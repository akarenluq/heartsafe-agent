"""Follow-up tracker: every escalation is acknowledged, reviewed, resolved and documented.

States: created -> acknowledged -> reviewed -> resolved. Only a clinician can move a
case forward or close it; the agent can only create cases, add evidence and send
reminders. Every action is written to an append-only audit log. Times are hours
since the start of monitoring (the simulation clock, or wall clock in a deployment).
"""
import json
import sqlite3

STATES = ["created", "acknowledged", "reviewed", "resolved"]
CLINICIAN_ACTIONS = {"acknowledge": ("created", "acknowledged"),
                     "review": ("acknowledged", "reviewed"),
                     "resolve": ("reviewed", "resolved")}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  case_id INTEGER PRIMARY KEY, patient_id TEXT, kind TEXT, severity TEXT, state TEXT,
  created_h REAL, ack_deadline_h REAL, resolve_deadline_h REAL,
  acknowledged_h REAL, reviewed_h REAL, resolved_h REAL,
  reminders INTEGER DEFAULT 0, assigned_to TEXT DEFAULT 'primary', review TEXT, outcome TEXT);
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
