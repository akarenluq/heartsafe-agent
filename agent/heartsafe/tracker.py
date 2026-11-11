"""Follow-up tracker: every escalation is acknowledged, reviewed, resolved and documented.

States: created -> acknowledged -> reviewed -> resolved. Only a clinician can move a
case forward or close it; the agent can only create cases, add evidence and send
reminders. Every action is written to an append-only audit log. Times are hours
since the start of monitoring (the simulation clock, or wall clock in a deployment).

Patient loop: after a clinician resolves a clinical case, the loop follows the
instruction to the patient: notified -> confirmed -> action_done -> closed (readings
improved) or reopened (they did not, or the action was not done). Only the patient,
their caregiver or a community health worker can record a confirmation or an action;
the agent sends the instruction, contacts helpers, checks the readings afterwards and
reopens a clinical case when needed. It never closes a clinical case.

Emergency cases (chest pain or fainting with a reliable abnormal ECG): the agent sends the
patient "If you feel worse, call 911." and opens the case; a clinician starts a video call
(acknowledge) and records the decision, "call_911" or "cancel", with a reason. The agent
has no way to call 911 or to record that decision.
"""
import json
import sqlite3

STATES = ["created", "acknowledged", "reviewed", "resolved"]
LOOP_STATES = ["notified", "confirmed", "action_done", "closed", "reopened"]
PATIENT_SIDE = {"patient", "caregiver", "community_health_worker"}
EMERGENCY_DECISIONS = {"call_911": "Video call: clinician called 911", "cancel": "Video call: alarm cancelled"}
CLINICIAN_ACTIONS = {"acknowledge": ("created", "acknowledged"),
                     "review": ("acknowledged", "reviewed"),
                     "resolve": ("reviewed", "resolved")}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  case_id INTEGER PRIMARY KEY, patient_id TEXT, kind TEXT, severity TEXT, state TEXT,
  created_h REAL, ack_deadline_h REAL, resolve_deadline_h REAL,
  acknowledged_h REAL, reviewed_h REAL, resolved_h REAL,
  reminders INTEGER DEFAULT 0, assigned_to TEXT DEFAULT 'primary', review TEXT, outcome TEXT);
CREATE TABLE IF NOT EXISTS loops (
  case_id INTEGER PRIMARY KEY, patient_id TEXT, state TEXT, instruction TEXT,
  notified_h REAL, confirm_deadline_h REAL, confirmed_h REAL, confirmed_by TEXT,
  action_done_h REAL, contacts INTEGER DEFAULT 0, outcome TEXT, closed_h REAL, reopened_case_id INTEGER);
CREATE TABLE IF NOT EXISTS audit (
  event_id INTEGER PRIMARY KEY, case_id INTEGER, t_h REAL, actor TEXT, role TEXT, action TEXT, detail TEXT);
"""


class PermissionDenied(Exception):
    pass


class Tracker:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def _log(self, case_id, t, actor, role, action, detail=None):
        self.db.execute("INSERT INTO audit (case_id, t_h, actor, role, action, detail) VALUES (?,?,?,?,?,?)",
                        (case_id, t, actor, role, action, json.dumps(detail) if detail is not None else None))

    def create(self, patient_id, kind, severity, t, ack_h, resolve_h, review):
        cur = self.db.execute(
            "INSERT INTO cases (patient_id, kind, severity, state, created_h, ack_deadline_h, resolve_deadline_h, review)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (patient_id, kind, severity, "created", t, t + ack_h, t + resolve_h, json.dumps(review)))
        self._log(cur.lastrowid, t, "heartsafe", "agent", "create", {"kind": kind, "severity": severity})
        return cur.lastrowid

    def open_case(self, patient_id):
        r = self.db.execute("SELECT * FROM cases WHERE patient_id=? AND state!='resolved' ORDER BY case_id DESC LIMIT 1",
                            (patient_id,)).fetchone()
        return dict(r) if r else None

    def add_evidence(self, case_id, t, detail):
        self._log(case_id, t, "heartsafe", "agent", "evidence", detail)

    def escalate_severity(self, case_id, t, severity, ack_deadline_h):
        self.db.execute("UPDATE cases SET severity=?, ack_deadline_h=MIN(ack_deadline_h, ?) WHERE case_id=?",
                        (severity, ack_deadline_h, case_id))
        self._log(case_id, t, "heartsafe", "agent", "severity_up", {"severity": severity})

    def log_llm(self, case_id, t, detail):
        """An LLM-written review was produced (or failed): logged, never a state change."""
        self._log(case_id, t, "heartsafe", "agent", "llm_review", detail)

    def remind(self, case_id, t, to="primary"):
        self.db.execute("UPDATE cases SET reminders=reminders+1, assigned_to=? WHERE case_id=?", (to, case_id))
        self._log(case_id, t, "heartsafe", "agent", "reminder", {"to": to})

    def act(self, case_id, action, t, actor, role, note=""):
        if role != "clinician":
            raise PermissionDenied(f"only a clinician can {action} a case")
        frm, to = CLINICIAN_ACTIONS[action]
        case = self.get(case_id)
        if case["state"] != frm:
            raise ValueError(f"case {case_id} is {case['state']}, cannot {action}")
        if action == "resolve" and not note.strip():
            raise ValueError("a resolution note is required")
        col = {"acknowledged": "acknowledged_h", "reviewed": "reviewed_h", "resolved": "resolved_h"}[to]
        self.db.execute(f"UPDATE cases SET state=?, {col}=? WHERE case_id=?", (to, t, case_id))
        if action == "resolve":
            self.db.execute("UPDATE cases SET outcome=? WHERE case_id=?", (note, case_id))
        self._log(case_id, t, actor, role, action, {"note": note} if note else None)

    def notify_patient(self, case_id, t, text):
        """A message the agent sends the patient straight away (no clinical decision in it)."""
        self._log(case_id, t, "heartsafe", "agent", "patient_message", {"text": text})

    def emergency_decision(self, case_id, t, actor, role, decision, reason):
        """After the video call the clinician records call 911 or cancel the alarm, with a reason."""
        if role != "clinician":
            raise PermissionDenied("only a clinician can decide to call 911 or cancel the alarm")
        if decision not in EMERGENCY_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(EMERGENCY_DECISIONS)}")
        if not (reason or "").strip():
            raise ValueError("a reason is required")
        case = self.get(case_id)
        if case["severity"] != "emergency":
            raise ValueError(f"case {case_id} is not an emergency")
        if case["state"] not in ("acknowledged", "reviewed"):
            raise ValueError(f"case {case_id} is {case['state']}: start the video call (acknowledge) first")
        outcome = f"{EMERGENCY_DECISIONS[decision]}. Reason: {reason}"
        self.db.execute("UPDATE cases SET state='resolved', resolved_h=?, outcome=? WHERE case_id=?",
                        (t, outcome, case_id))
        self._log(case_id, t, actor, role, "emergency_decision", {"decision": decision, "reason": reason})

    # ---- patient loop ------------------------------------------------------
    def start_loop(self, case_id, t, instruction, confirm_deadline_h):
        case = self.get(case_id)
        if case["state"] != "resolved":
            raise ValueError("the patient loop starts after a clinician resolves the case")
        self.db.execute("INSERT INTO loops (case_id, patient_id, state, instruction, notified_h, confirm_deadline_h)"
                        " VALUES (?,?,?,?,?,?)", (case_id, case["patient_id"], "notified", instruction, t,
                                                  t + confirm_deadline_h))
        self._log(case_id, t, "heartsafe", "agent", "patient_notified", {"instruction": instruction})

    def contact(self, case_id, t, to):
        """The agent follows up with the patient, caregiver or community health worker."""
        self.db.execute("UPDATE loops SET contacts=contacts+1 WHERE case_id=?", (case_id,))
        self._log(case_id, t, "heartsafe", "agent", "patient_contact", {"to": to})

    def patient_step(self, case_id, step, t, actor, role):
        """confirm or action_done, recorded by the patient side only."""
        if role not in PATIENT_SIDE:
            raise PermissionDenied(f"only the patient, a caregiver or a community health worker can {step}")
        frm, to, col = {"confirm": ("notified", "confirmed", "confirmed_h"),
                        "action_done": ("confirmed", "action_done", "action_done_h")}[step]
        loop = self.loop(case_id)
        if loop["state"] != frm:
            raise ValueError(f"loop {case_id} is {loop['state']}, cannot {step}")
        self.db.execute(f"UPDATE loops SET state=?, {col}=? WHERE case_id=?", (to, t, case_id))
        if step == "confirm":
            self.db.execute("UPDATE loops SET confirmed_by=? WHERE case_id=?", (role, case_id))
        self._log(case_id, t, actor, role, f"patient_{step}")

    def close_loop(self, case_id, t, outcome, reopened_case_id=None):
        """Agent: readings checked after the action. closed = improved; reopened = new clinical case."""
        loop = self.loop(case_id)
        if loop["state"] in ("closed", "reopened"):
            raise ValueError(f"loop {case_id} is already {loop['state']}")
        if reopened_case_id is None and loop["state"] != "action_done":
            raise ValueError("a loop closes only after the patient side records the action")
        state = "reopened" if reopened_case_id is not None else "closed"
        self.db.execute("UPDATE loops SET state=?, outcome=?, closed_h=?, reopened_case_id=? WHERE case_id=?",
                        (state, outcome, t, reopened_case_id, case_id))
        self._log(case_id, t, "heartsafe", "agent", f"loop_{state}",
                  {"outcome": outcome, "reopened_case_id": reopened_case_id})

    def loop(self, case_id):
        r = self.db.execute("SELECT * FROM loops WHERE case_id=?", (case_id,)).fetchone()
        return dict(r) if r else None

    def loops(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM loops ORDER BY case_id")]

    def due_reminders(self, now):
        """Cases past a deadline that still wait for the next clinician step."""
        rows = self.db.execute(
            "SELECT * FROM cases WHERE (state='created' AND ack_deadline_h<?) OR "
            "(state!='resolved' AND resolve_deadline_h<?)", (now, now)).fetchall()
        return [dict(r) for r in rows]

    def get(self, case_id):
        return dict(self.db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())

    def cases(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM cases ORDER BY case_id")]

    def audit(self, case_id=None):
        q, a = ("SELECT * FROM audit WHERE case_id=? ORDER BY t_h, event_id", (case_id,)) if case_id else \
            ("SELECT * FROM audit ORDER BY t_h, event_id", ())
        return [dict(r) for r in self.db.execute(q, a)]

    def commit(self):
        self.db.commit()

    def save_to(self, path):
        disk = sqlite3.connect(str(path))
        self.db.backup(disk)
        disk.close()
