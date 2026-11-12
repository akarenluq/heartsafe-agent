"""Self-care the patient logs every day: last night's sleep and breathing-exercise sessions.

Sleep is compared with the patient's own usual sleep (the median of their earlier nights),
not with a fixed number of hours. Worse sleep alone is never a clinical alert: the agent
opens a low-priority check-in call for the nurse and shows the sleep and breathing lines
in every review. The breathing plan (how many sessions a day, which exercise) comes from
the care team; HeartSafe only counts the sessions the patient logs.

SelfCareLog stores what the patient side logs in the app (the "Sleep and breathing" page).
"""
import json
import sqlite3

import numpy as np

from .tracker import PATIENT_SIDE, PermissionDenied

KEPT_AWAKE = ["pain", "cough", "worry or low mood", "going to the bathroom", "something else"]
# Two answers in the sleep log are symptoms the platform already alerts on (caution level).
SLEEP_SYMPTOMS = {"propped_up": "orthopnea", "woke_breathless": "pnd"}

LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS selfcare (
  entry_id INTEGER PRIMARY KEY, patient_id TEXT, kind TEXT, day INTEGER, logged_h REAL,
  actor TEXT, role TEXT, detail TEXT);
"""


class SelfCareLog:
    """Sleep entries and breathing sessions logged by the patient side. Append-only: the last sleep
    entry of a night is the one that counts; every breathing entry is one session. Kept in its own
    file, so re-running the cohort (which rebuilds the case database) does not erase it."""

    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(LOG_SCHEMA)

    def _add(self, patient_id, kind, day, t, actor, role, detail):
        if role not in PATIENT_SIDE:
            raise PermissionDenied("only the patient, a caregiver or a community health worker can log "
                                   "sleep or breathing exercises")
        cur = self.db.execute("INSERT INTO selfcare (patient_id, kind, day, logged_h, actor, role, detail)"
                              " VALUES (?,?,?,?,?,?,?)",
                              (patient_id, kind, int(day), t, actor, role, json.dumps(detail)))
        self.db.commit()
        return cur.lastrowid

    def log_sleep(self, patient_id, day, t, actor, role, hours, awakenings=0, propped_up=False,
                  woke_breathless=False, kept_awake=()):
        """Last night's sleep, logged on the morning of `day`."""
        if not 0 <= hours <= 24:
            raise ValueError("hours slept must be between 0 and 24")
        if awakenings < 0:
            raise ValueError("times woken cannot be negative")
        return self._add(patient_id, "sleep", day, t, actor, role, {
            "hours": float(hours), "awakenings": int(awakenings), "propped_up": bool(propped_up),
            "woke_breathless": bool(woke_breathless), "kept_awake": list(kept_awake)})

    def log_breathing(self, patient_id, day, t, actor, role, exercise, minutes):
        """One finished breathing-exercise session."""
        if minutes <= 0:
            raise ValueError("a session has a length in minutes")
        return self._add(patient_id, "breathing", day, t, actor, role,
                         {"exercise": exercise, "minutes": float(minutes)})

    def entries(self, patient_id, kind=None):
        q, a = "SELECT * FROM selfcare WHERE patient_id=?", [patient_id]
        if kind:
            q, a = q + " AND kind=?", a + [kind]
        return [{**dict(r), "detail": json.loads(r["detail"])}
                for r in self.db.execute(q + " ORDER BY day, entry_id", a)]

    def days(self, patient_id):
        """{day: {"day", "sleep_hours", "sleep_awakenings", "breathing_sessions", ...}} from the log."""
        out = {}
        for e in self.entries(patient_id):
            d = out.setdefault(e["day"], {"day": e["day"]})
            if e["kind"] == "sleep":
                x = e["detail"]
                d.update(sleep_hours=x["hours"], sleep_awakenings=x["awakenings"], propped_up=x["propped_up"],
                         woke_breathless=x["woke_breathless"], kept_awake=x["kept_awake"])
            else:
                d["breathing_sessions"] = d.get("breathing_sessions", 0) + 1
        return out


def with_logged(hist, logged):
    """The platform's history (dicts with "day", oldest first) with the days logged in the app: a
    logged sleep entry replaces that night, logged breathing sessions are added to that day."""
    by_day = {h["day"]: dict(h) for h in hist}
    for d, x in logged.items():
        h = by_day.setdefault(d, {"day": d})
        n = x.get("breathing_sessions")
        if n:
            h["breathing_sessions"] = (h["breathing_sessions"] if _num(h.get("breathing_sessions")) else 0) + n
        h.update({k: v for k, v in x.items() if k != "breathing_sessions"})
    return [by_day[d] for d in sorted(by_day)]


def _num(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def nights(hist, today=None):
    """(day, hours, times woken) for every day with a sleep entry, oldest first."""
    rows = list(hist) + ([today] if today else [])
    return [(h["day"], float(h["sleep_hours"]),
             float(h["sleep_awakenings"]) if _num(h.get("sleep_awakenings")) else None)
            for h in rows if _num(h.get("sleep_hours"))]


def sleep_status(hist, today, cfg):
    """The last nights against the patient's usual sleep. None when there is no sleep entry."""
    s = (cfg.get("selfcare") or {}).get("sleep")
    n = nights(hist, today)
    if not s or not n:
        return None
    w = s["window_nights"]
    recent, earlier = n[-w:], n[:-w]
    out = {"nights": len(recent), "recent_h": round(float(np.mean([h for _, h, _ in recent])), 1),
           "usual_h": None, "worse": False, "text": None}
    woke = [a for _, _, a in recent if a is not None]
    if woke:
        out["recent_awakenings"] = round(float(np.mean(woke)), 1)
    if len(recent) < w or len(earlier) < s["min_usual_nights"]:
        return out
    usual = float(np.median([h for _, h, _ in earlier]))
    out["usual_h"] = round(usual, 1)
    usual_woke = [a for _, _, a in earlier if a is not None]
    if usual_woke:
        out["usual_awakenings"] = round(float(np.median(usual_woke)), 1)
    every_night = all(usual - h >= s["each_night_h"] for _, h, _ in recent)
    avg = float(np.mean([h for _, h, _ in recent]))
    if every_night and (usual - avg >= s["drop_h"] or avg < s["short_h"]):
        out["worse"] = True
        out["text"] = (f"sleep worse than usual: {avg:.1f} h a night over the last {w} nights "
                       f"(usual {usual:.1f} h)")
        if woke and usual_woke and out["recent_awakenings"] > out["usual_awakenings"]:
            out["text"] += (f", woke {out['recent_awakenings']:g} times a night "
                            f"(usual {out['usual_awakenings']:g})")
    return out


def breathing_status(hist, today, cfg, days=7):
    """Breathing-exercise sessions a day over the last week against the care team's plan."""
    b = (cfg.get("selfcare") or {}).get("breathing")
    rows = [h for h in list(hist) + ([today] if today else []) if _num(h.get("breathing_sessions"))][-days:]
    if not b or not rows:
        return None
    return {"plan": b["sessions_per_day"], "days": len(rows),
            "per_day": round(float(np.mean([h["breathing_sessions"] for h in rows])), 1)}


def summary(hist, today, cfg):
    """What a review shows about self-care; None when the platform has no such entries."""
    out = {"sleep": sleep_status(hist, today, cfg), "breathing": breathing_status(hist, today, cfg)}
    return out if out["sleep"] or out["breathing"] else None


def describe(sc):
    """The self-care lines of a review, in plain words."""
    if not sc:
        return None
    parts = []
    s, b = sc.get("sleep"), sc.get("breathing")
    if s:
        if s.get("worse"):
            parts.append(s["text"][0].upper() + s["text"][1:] + ".")
        elif s.get("usual_h") is not None:
            parts.append(f"Sleep {s['recent_h']:.1f} h a night over the last {s['nights']} nights "
                         f"(usual {s['usual_h']:.1f} h).")
        else:
            parts.append(f"Sleep {s['recent_h']:.1f} h a night over the last {s['nights']} night(s); "
                         "not enough nights yet to know the usual.")
    if b:
        parts.append(f"Breathing exercises: {b['per_day']:g} of {b['plan']} planned sessions a day "
                     f"over the last {b['days']} days.")
    return " ".join(parts) if parts else None
