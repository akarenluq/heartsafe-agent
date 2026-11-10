"""Live check of the Claude review orchestrator on a few real cases from data/heartsafe.db.

Usage (from the agent/ folder, after a run of heartsafe.run):
  set ANTHROPIC_API_KEY=...            (PowerShell: $env:ANTHROPIC_API_KEY="...")
  .venv/Scripts/python.exe llm_check.py            # 5 cases, the model in config.yaml
  .venv/Scripts/python.exe llm_check.py --n 3

Works on a copy of the database, so the dashboard's audit log is not touched. For each case it
prints the tools Claude called and the review, and checks that: the review came from the model
(not the template), submit_review was called, and every number in the review appears in the
case facts the tools returned (a simple check for invented numbers). Cost: a few cents per case.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from heartsafe import DATA, load_config
from heartsafe.orchestrator import _tool_results, orchestrate
from heartsafe.review import render
from heartsafe.tracker import Tracker

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _known(x, known):
    """A number in the review matches a fact when equal at the precision it is shown with (or a %)."""
    d = len(x.split(".")[1]) if "." in x else 0
    v, tol = float(x), 0.5 * 10 ** -d + 1e-9
    return v <= 3 or any(abs(k - v) <= tol or abs(100 * k - v) <= tol for k in known)


def pick_cases(cases, n):
    """One of each kind, starting with the demo patient's critical case, then other clinical cases."""
    by = sorted(cases, key=lambda c: (c["patient_id"] != "DEMO-MARY", c["case_id"]))
    out, kinds = [], set()
    for c in by:
        k = (c["kind"], c["severity"])
        if k not in kinds:
            out.append(c)
            kinds.add(k)
    out += [c for c in by if c not in out and c["kind"] == "clinical"]
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model")
    args = ap.parse_args()
    model = args.model or load_config()["review"]["llm_model"]
    tmp = Path(tempfile.mkdtemp()) / "heartsafe.db"
    shutil.copy(DATA / "heartsafe.db", tmp)
    tr = Tracker(tmp)
    ok = 0
    for case in pick_cases(tr.cases(), args.n):
        review, facts = _tool_results(case, tr)
        source = json.dumps(facts, default=str) + render(review)  # the template's own wording counts as known
        known = [float(x) for x in NUM.findall(source)] + [case["case_id"]]
        out = orchestrate(case, tr, model, now_h=case["created_h"])
        invented = sorted({x for x in NUM.findall(out["text"]) if not _known(x, known)})
        good = out["source"] != "template" and "submit_review" in out["tool_calls"] and not invented
        ok += good
        print("=" * 90)
        print(f"case {case['case_id']} {case['kind']}/{case['severity']} {case['patient_id']}: "
              f"source={out['source']} tools={out['tool_calls']}")
        print(out["text"])
        why = ["template used (no API key, or the call failed: see the note above)"] if out["source"] == "template" \
            else []
        why += [f"numbers not in the facts: {invented}"] if invented else []
        print(f"-> {'PASS' if good else 'CHECK'}" + (f" ({'; '.join(why)})" if why else ""))
    print("=" * 90)
    print(f"{ok} of {min(args.n, len(tr.cases()))} reviews passed the checks")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
