"""Follow-up watch: every few hours, check where each case and patient loop stands.

It answers, without an LLM, what a nurse needs to know about follow-up: is the case
acknowledged, is a deadline close or missed, how many reminders went out, and has the
patient side confirmed and acted on the care team's instruction. `Watcher.tick` runs
it on a fixed interval (`followup.watch_every_h` in config.yaml); with `act=True` it
also sends a reminder for a missed deadline or contacts the next helper for a missed
patient confirmation. Like the rest of the agent it never acknowledges, resolves or
closes anything: those stay with the clinical team and the patient side.

Live use (from the agent/ folder), one check every 2 hours on the tracker database:
  python -m heartsafe.watch                 # one check now (simulation clock), report only
  python -m heartsafe.watch --act --loop    # keep checking every watch_every_h hours
"""
import argparse
import json
import time

from . import DATA, load_config
from .tracker import Tracker

HELPERS = ["caregiver", "community_health_worker"]
RANK = {"emergency": -1, "critical": 0, "caution": 1, "low": 2}


def _h(x):
    """Hours as readable text: 3.5 -> '3.5 h', 50 -> '2.1 days'."""
    return f"{x:.1f} h" if abs(x) < 48 else f"{x / 24:.1f} days"


def _last(audit, action):
    ts = [a["t_h"] for a in audit if a["action"] == action]
    return max(ts) if ts else None


def followup_status(case, tracker, now):
    """Where one case stands at time `now` (hours): a dict and plain-English lines for the review."""
    audit = [a for a in tracker.audit(case["case_id"]) if a["t_h"] <= now]
    done = {a["action"]: a["t_h"] for a in audit if a["role"] == "clinician"}
    if "emergency_decision" in done:  # an emergency is closed by the video-call decision (call 911 or cancel)
        done.setdefault("resolve", done["emergency_decision"])
    state = ("resolved" if "resolve" in done else "reviewed" if "review" in done else
             "acknowledged" if "acknowledge" in done else "created")
    reminders = [a for a in audit if a["action"] == "reminder"]
    issues, lines = [], []

    if state == "created":
        left = case["ack_deadline_h"] - now
        if left < 0:
            issues.append("ack_overdue")
            lines.append(f"Not acknowledged; the acknowledgement deadline passed {_h(-left)} ago.")
        else:
            lines.append(f"Waiting for acknowledgement; deadline in {_h(left)}.")
    elif state != "resolved":
        lines.append(f"{state.capitalize()} {_h(now - done[{'acknowledged': 'acknowledge', 'reviewed': 'review'}[state]])}"
                     " ago; waiting for resolution.")
        if case["resolve_deadline_h"] < now:
            issues.append("resolve_overdue")
            lines.append(f"The resolution deadline passed {_h(now - case['resolve_deadline_h'])} ago.")
    else:
        note = next((json.loads(a["detail"]).get("note") for a in audit if a["action"] == "resolve" and a["detail"]),
                    "") or (case.get("outcome") or "" if "emergency_decision" in done else "")
        lines.append(f"Resolved {_h(now - done['resolve'])} ago" + (f": {note}" if note else "."))
    if reminders:
        to = json.loads(reminders[-1]["detail"] or "{}").get("to", "primary")
        lines.append(f"{len(reminders)} reminder(s) sent; the last one {_h(now - reminders[-1]['t_h'])} ago, "
                     f"to the {'backup' if to == 'backup' else 'primary'} nurse.")

    loop = tracker.loop(case["case_id"])
    loop_state = None
    if loop and loop["notified_h"] is not None and loop["notified_h"] <= now:
        steps = {a["action"]: a for a in audit}
        contacts = [json.loads(a["detail"] or "{}").get("to") for a in audit if a["action"] == "patient_contact"]
        if "loop_closed" in steps or "loop_reopened" in steps:
            ev = steps.get("loop_closed") or steps["loop_reopened"]
            loop_state = "closed" if "loop_closed" in steps else "reopened"
            lines.append(f"Patient loop {loop_state}: {json.loads(ev['detail'] or '{}').get('outcome', '')}.")
        elif "patient_action_done" in steps:
            loop_state = "action_done"
            lines.append(f"The {steps['patient_action_done']['role'].replace('_', ' ')} recorded the action "
                         f"{_h(now - steps['patient_action_done']['t_h'])} ago; readings are checked after it.")
        elif "patient_confirm" in steps:
            loop_state = "confirmed"
            lines.append(f"Instruction confirmed by the {steps['patient_confirm']['role'].replace('_', ' ')} "
                         f"{_h(now - steps['patient_confirm']['t_h'])} ago; waiting for the action.")
        else:
            loop_state = "notified"
            left = loop["confirm_deadline_h"] - now
            if left < 0:
                issues.append("patient_not_confirmed")
                lines.append(f"Instruction sent {_h(now - loop['notified_h'])} ago and not confirmed; "
                             f"the confirmation deadline passed {_h(-left)} ago.")
            else:
                lines.append(f"Instruction sent to the patient; confirmation due in {_h(left)}.")
        if contacts:
            lines.append("Helpers contacted: " + ", ".join(c.replace("_", " ") for c in contacts) + ".")
    return {"case_id": case["case_id"], "patient_id": case["patient_id"], "kind": case["kind"],
            "severity": case["severity"], "state": state, "loop_state": loop_state,
            "reminders": len(reminders), "issues": issues, "lines": lines}


def watch(tracker, now):
    """Every case or patient loop that needs a follow-up action now, most urgent first."""
    out = []
    for c in tracker.cases():
        if c["created_h"] > now:
            continue
        s = followup_status(c, tracker, now)
        if s["issues"]:
            out.append(s)
    return sorted(out, key=lambda s: (RANK.get(s["severity"], 3), s["case_id"]))


class Watcher:
    """Runs `watch` once every `every_h` hours of the monitoring clock."""

    def __init__(self, tracker, cfg, every_h=None):
        self.tr = tracker
        self.every_h = every_h or cfg["followup"]["watch_every_h"]
        self.last_h = None

    def due(self, now):
        return self.last_h is None or now - self.last_h >= self.every_h

    def tick(self, now, act=False):
        """Check now if a check is due. act=True: remind the nurse or contact the next helper (no state change)."""
        if not self.due(now):
            return None
        self.last_h = now
        found = watch(self.tr, now)
        if act:
            for s in found:
                audit = self.tr.audit(s["case_id"])
                if {"ack_overdue", "resolve_overdue"} & set(s["issues"]):
                    last = _last(audit, "reminder")
                    if last is None or now - last >= self.every_h:
                        self.tr.remind(s["case_id"], now, "backup" if s["reminders"] else "primary")
                if "patient_not_confirmed" in s["issues"]:
                    asked = [json.loads(a["detail"] or "{}").get("to") for a in audit if a["action"] == "patient_contact"]
                    nxt = next((h for h in HELPERS if h not in asked), None)
                    last = _last(audit, "patient_contact")
                    if nxt and (last is None or now - last >= self.every_h):
                        self.tr.contact(s["case_id"], now, nxt)
            self.tr.commit()
        return found


def report(found, now, every_h):
    head = f"Follow-up watch at day {now / 24:.1f} (every {every_h:g} h): {len(found)} item(s) need action"
    rows = [f"- case {s['case_id']} {s['patient_id']} ({s['kind']}, {s['severity']}): {' '.join(s['lines'])}"
            for s in found]
    return "\n".join([head] + rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", type=float, help="hours since the start of monitoring (default: end of the last run)")
    ap.add_argument("--every", type=float, help="hours between checks (default: followup.watch_every_h)")
    ap.add_argument("--act", action="store_true", help="send reminders and helper contacts, not only report")
    ap.add_argument("--loop", action="store_true", help="keep checking (wall clock) instead of checking once")
    args = ap.parse_args(argv)
    cfg = load_config()
    w = Watcher(Tracker(DATA / "heartsafe.db"), cfg, args.every)
    now = args.now if args.now is not None else json.load(open(DATA / "run_info.json"))["horizon_h"]
    start = time.time()
    while True:
        t = now + (time.time() - start) / 3600
        print(report(w.tick(t, act=args.act), t, w.every_h), flush=True)
        if not args.loop:
            break
        time.sleep(w.every_h * 3600)


if __name__ == "__main__":
    main()
