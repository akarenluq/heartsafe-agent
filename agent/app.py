"""HeartSafe dashboard. Run from the agent/ folder:  streamlit run app.py"""
import json

import altair as alt
import pandas as pd
import streamlit as st

import subprocess
import sys
from pathlib import Path

from heartsafe import DATA, load_config
from heartsafe.orchestrator import orchestrate
from heartsafe.review import render
from heartsafe.tracker import PermissionDenied, Tracker

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="HeartSafe", page_icon=":anatomical_heart:", layout="wide")
if not (DATA / "heartsafe.db").exists():
    st.error("No results yet. In a terminal in the agent folder run:  python -m heartsafe.run")
    st.stop()

cfg = load_config()


@st.cache_data
def load(name):
    p = DATA / name
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return json.load(open(p)) if p.exists() else None


@st.cache_resource
def tracker():
    return Tracker(DATA / "heartsafe.db")


tr = tracker()
feed, patients, decisions = load("feed.csv"), load("patients.csv"), load("decisions.csv")
now = load("run_info.json")["horizon_h"]
SEV_ORDER = {"critical": 0, "caution": 1, "low": 2}

st.sidebar.title("HeartSafe")
st.sidebar.caption("Trust and coordination layer on top of remote monitoring platforms (e.g. Medly)")
role = st.sidebar.selectbox("Signed in as", ["clinician", "administrator"])
user = st.sidebar.text_input("Name", "nurse_demo" if role == "clinician" else "admin_demo")
page = st.sidebar.radio("View", ["Patient journey", "Clinician queue", "Equity audit", "Results vs platform alone",
                                 "Model reliability", "How it works"])
st.sidebar.caption(f"Simulation clock: day {now / 24:.0f}. Platform data here is simulated (modelled on Medly, which is not open source).")


def vitals_chart(pid, upto_day=None, window=None):
    f = feed[(feed.patient_id == pid) & feed.submitted]
    if upto_day is not None:
        f = f[(f.day <= upto_day) & (f.day > upto_day - (window or 21))]
    d = decisions[decisions.patient_id == pid]
    base = alt.Chart(f).mark_line(point=True).encode(x=alt.X("day:Q", title="Day"),
                                                     y=alt.Y("weight_kg:Q", title="Weight (kg)", scale=alt.Scale(zero=False)))
    marks = alt.Chart(d.merge(f[["day", "weight_kg"]], on="day")).mark_point(size=160, filled=True).encode(
        x="day:Q", y="weight_kg:Q", color=alt.Color("decision:N", title="HeartSafe action"),
        tooltip=["day", "decision", "severity", "alert_rules", "flags"])
    return (base + marks).properties(height=260)


def show_case(case):
    review = json.loads(case["review"])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(render(review))
        if st.button("Explain with AI agent", key=f"llm{case['case_id']}",
                     help="Claude calls read-only tools over this case and writes the review. "
                          "The decision and deadlines stay with the policy and the clinical team."):
            with st.spinner("Agent gathering facts..."):
                out = orchestrate(case, tr, cfg["review"]["llm_model"], now_h=now)
            st.info(out["text"])
            st.caption(f"source: {out['source']}" + (f" | tools called: {', '.join(out['tool_calls'])}"
                                                      if out["tool_calls"] else ""))
    with c2:
        st.altair_chart(vitals_chart(case["patient_id"], review["day"]), width="stretch")
        if review.get("ecg"):
            e = review["ecg"]
            msg = (f"ECG: {'abnormal' if e['abnormal'] else 'no abnormality'} (p={e['prob']:.2f}), "
                   f"quality {e['quality']:.2f}")
            (st.success if e["reliable"] else st.warning)(
                msg + ("" if e["reliable"] else f"  \nNot used: {', '.join(e.get('reasons') or [])}"))
        if review.get("mri"):
            m = review["mri"]
            msg = f"MRI{' (' + m['scanner'] + ' scanner)' if m.get('scanner') else ''}: LVEF {m['lvef']}%"
            (st.success if m["reliable"] else st.warning)(
                msg + ("" if m["reliable"] else f"  \nFlagged: {', '.join(m.get('reasons') or m['flags'])}")
                + ("".join(f"  \n{c}" for c in m.get("caveats") or [])))
    loop = tr.loop(case["case_id"]) if hasattr(tr, "loop") else None
    if loop:
        steps = [("Instruction sent", loop["notified_h"]), ("Confirmed" + (f" by {loop['confirmed_by'].replace('_', ' ')}"
                  if loop["confirmed_by"] else ""), loop["confirmed_h"]), ("Action done", loop["action_done_h"]),
                 ({"closed": "Readings checked: improved", "reopened": "Reopened"}.get(loop["state"], "Readings check"),
                  loop["closed_h"])]
        st.markdown("**Patient loop** (after the clinician's decision)  \n" + "  \n".join(
            f"{'✓' if t is not None else '·'} {name}" + (f", day {t / 24:.1f}" if t is not None else "")
            for name, t in steps) + (f"  \nHelper contacts: {loop['contacts']}" if loop["contacts"] else "")
            + (f"  \nReason: {loop['outcome']}" if loop["state"] == "reopened" else ""))
    st.markdown("**Audit log**")
    log = pd.DataFrame(tr.audit(case["case_id"]))
    log["day"] = (log.t_h / 24).round(2)
    st.dataframe(log[["day", "actor", "role", "action", "detail"]], hide_index=True, width="stretch")
    nxt = {"created": "acknowledge", "acknowledged": "review", "reviewed": "resolve"}.get(case["state"])
    if nxt:
        note = st.text_area("Note (required to resolve)", key=f"note{case['case_id']}") if nxt == "resolve" else ""
        if st.button(nxt.capitalize(), type="primary", key=f"act{case['case_id']}"):
            try:
                tr.act(case["case_id"], nxt, now, user, role, note)
                tr.commit()
                st.success(f"Case {case['case_id']}: {nxt} recorded by {user}.")
                st.rerun()
            except PermissionDenied as e:
                st.error(f"Not allowed: {e}. HeartSafe never closes a case on its own.")
            except ValueError as e:
                st.error(str(e))


if page == "Patient journey":
    st.header("Patient journey")
    pid = st.selectbox("Patient", ["DEMO-MARY"] + sorted(p for p in decisions.patient_id.unique() if p != "DEMO-MARY"))
    p = patients[patients.patient_id == pid].iloc[0]
    extra = ""
    if "nyha" in p:
        extra = (f" · NYHA {p.nyha}, discharged {p.days_since_discharge} days before enrollment, "
                 f"caregiver: {'yes' if p.caregiver else 'no'}")
    st.caption(f"{p.age} y, {p.sex}, {p.remoteness} community, {p.device.replace('_', ' ')}, "
               f"ECG device: {'yes' if p.ecg_kit else 'no'}{extra}")
    st.altair_chart(vitals_chart(pid), width="stretch")
    d = decisions[decisions.patient_id == pid][["day", "alert_rules", "decision", "severity", "flags"]]
    st.markdown("**Every platform alert and what HeartSafe did with it**")
    st.dataframe(d, hide_index=True, width="stretch")
    cases = pd.DataFrame(tr.cases())
    for _, c in cases[cases.patient_id == pid].iterrows():
        with st.expander(f"Case {c.case_id}: {c.kind}, {c.severity}, {c.state}, reminders {c.reminders}"):
            show_case(c.to_dict())

elif page == "Clinician queue":
    st.header("Clinician queue")
    cases = pd.DataFrame(tr.cases())
    open_ = cases[cases.state != "resolved"].copy()
    open_["overdue"] = (open_.state == "created") & (open_.ack_deadline_h < now)
    open_["created_day"] = (open_.created_h / 24).round(1)
    open_ = open_.sort_values(["overdue", "severity"], ascending=[False, True],
                              key=lambda s: s.map(SEV_ORDER) if s.name == "severity" else s)
    m1, m2, m3 = st.columns(3)
    m1.metric("Open cases", len(open_))
    m2.metric("Overdue acknowledgements", int(open_.overdue.sum()))
    m3.metric("Critical", int((open_.severity == "critical").sum()))
    st.dataframe(open_[["case_id", "patient_id", "kind", "severity", "state", "created_day", "overdue",
                        "reminders", "assigned_to"]], hide_index=True, width="stretch")
    if len(open_):
        cid = st.selectbox("Open a case", open_.case_id.tolist())
        show_case(tr.get(int(cid)))

elif page == "Equity audit":
    st.header("Equity audit")
    st.caption("Observable metrics only (no ground truth). Red = the subgroup's whole 95% CI is worse than overall.")
    eq = load("equity.csv")
    metric = st.selectbox("Metric", eq.metric.unique(), format_func=lambda m: eq[eq.metric == m].label.iloc[0])
    e = eq[eq.metric == metric].copy()
    e["row"] = e.group + " = " + e.value
    overall = e[e.group == "overall"]["mean"].iloc[0]
    base = alt.Chart(e).encode(y=alt.Y("row:N", sort=None, title=None))
    chart = (base.mark_rule().encode(x=alt.X("ci_low:Q", title=e.label.iloc[0]), x2="ci_high:Q",
                                     color=alt.condition("datum.gap", alt.value("#c0392b"), alt.value("#555")))
             + base.mark_point(filled=True, size=70).encode(
                 x="mean:Q", color=alt.condition("datum.gap", alt.value("#c0392b"), alt.value("#555")),
                 tooltip=["row", "n", alt.Tooltip("mean:Q", format=".2f"), alt.Tooltip("ci_low:Q", format=".2f"),
                          alt.Tooltip("ci_high:Q", format=".2f")])
             + alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"))
    st.altair_chart(chart.properties(height=28 * len(e) + 40), width="stretch")

    st.subheader("Safeguard proposals")
    props = load("proposals.json") or []
    sg_path = DATA / "safeguards.json"
    approved = json.load(open(sg_path)) if sg_path.exists() else []
    approved_ids = {s["id"] for s in approved if s.get("approved")}
    for p in props:
        with st.container(border=True):
            st.markdown(f"**{p['id']}**  \n{p['evidence']}  \nEffect: {p['effect']}")
            if p["type"] == "recommendation":
                st.caption("Program recommendation; not applied automatically.")
            elif p["id"] in approved_ids:
                st.success(f"Approved by {next(s.get('approved_by', '?') for s in approved if s['id'] == p['id'])}.")
            elif role == "administrator":
                if st.button("Approve safeguard", key=p["id"]):
                    approved.append({**p, "approved": True, "approved_by": user})
                    json.dump(approved, open(sg_path, "w"), indent=2)
                    st.rerun()
            else:
                st.caption("Needs administrator approval.")

    applied = set((load("evaluation.json") or {}).get("safeguards_applied", []))
    pending = approved_ids - applied
    if pending and role == "administrator":
        st.warning(f"Approved but not applied yet: {', '.join(sorted(pending))}")
        if st.button("Apply approved safeguards now (re-runs the cohort, about 1 minute)", type="primary"):
            with st.spinner("Re-running HeartSafe on the cohort with the approved safeguards..."):
                r = subprocess.run([sys.executable, "-m", "heartsafe.run"], cwd=Path(__file__).parent,
                                   capture_output=True, text=True)
            if r.returncode == 0:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(r.stderr[-2000:])
    effect = (load("evaluation.json") or {}).get("safeguard_effect")
    if effect and "note" not in effect:
        st.subheader("What the approved safeguards changed")
        c = st.columns(3)
        c[0].metric("Extra clinical cases / 100 patient-months", f"{effect['extra_clinical_cases_per_100_pm']:+.1f}")
        c[1].metric("Extra false alarms / 100 patient-months", f"{effect['extra_false_alarms_per_100_pm']:+.1f}")
        c[2].metric("Extra reminders sent", f"{effect['extra_reminders']:+d}")
        st.dataframe(pd.DataFrame(effect["targeted_groups"]), hide_index=True, width="stretch")
        st.caption("Before = the last run with no approved safeguards. Same cohort and seed.")

elif page == "Results vs platform alone":
    st.header("HeartSafe vs monitoring platform alone (simulated cohort)")
    r = load("evaluation.json")
    te = r["true_episodes"]
    scr = load("screening.json")
    if scr:
        st.caption(f"Enrollment: {scr['enrolled']} of {scr['candidates']} screened candidates met the inclusion "
                   "criteria. Most common reasons not eligible: "
                   + "; ".join(f"{k} ({v})" for k, v in list(scr["reasons"].items())[:4]) + ".")
    c = st.columns(4)
    c[0].metric("Faulty-reading alerts stopped", f"{r['fault_driven_alerts']['stopped_by_heartsafe']:.0%}",
                help=f"{r['fault_driven_alerts']['n']} alerts caused by device or entry errors; the platform alone forwards all")
    c[1].metric("Mean hours, first alert to acknowledgement",
                f"{te['mean_hours_first_alert_to_ack']['heartsafe']}",
                f"{te['mean_hours_first_alert_to_ack']['heartsafe'] - te['mean_hours_first_alert_to_ack']['platform_alone']:+.1f} vs platform alone",
                delta_color="inverse")
    c[2].metric("True episodes acknowledged within 24 h", f"{te['acknowledged_within_24h']['heartsafe']:.0%}",
                f"{(te['acknowledged_within_24h']['heartsafe'] - te['acknowledged_within_24h']['platform_alone']) * 100:+.0f} pts")
    c[3].metric("Alert items never acknowledged (platform alone)", r["followup"]["platform_alone_alert_items_never_acknowledged"],
                help="With HeartSafe every case is tracked to acknowledgement (reminders, backup routing)")
    by = pd.DataFrame([{"remoteness": k, "arm": arm, "hours": v[f"{key}_mean"]}
                       for k, v in te["hours_to_ack_by_remoteness"].items()
                       for arm, key in (("HeartSafe", "heartsafe"), ("Platform alone", "platform_alone"))])
    st.altair_chart(alt.Chart(by).mark_bar().encode(
        x=alt.X("remoteness:N", sort=["urban", "rural", "remote"], title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="arm:N", y=alt.Y("hours:Q", title="Mean hours to acknowledge"),
        color=alt.Color("arm:N", title=None)).properties(height=280), width="stretch")

    st.subheader("Follow-up")
    res, stl = r.get("resolution", {}), r.get("stalled_cases", {})
    rows = []
    if res:
        rows += [{"Measure": "Escalations resolved within deadline",
                  "HeartSafe": f"{res['heartsafe']['resolved_within_deadline']:.0%}",
                  "Platform alone": f"{res['platform_alone']['resolved_within_deadline']:.0%}"},
                 {"Measure": "Median hours to resolution", "HeartSafe": res["heartsafe"]["median_hours_to_resolution"],
                  "Platform alone": res["platform_alone"]["median_hours_to_resolution"]}]
    if stl:
        rows.append({"Measure": "Stalled cases found",
                     "HeartSafe": f"{stl['heartsafe']['flagged_by_reminder']} of {stl['heartsafe']['simulated_to_stall']}",
                     "Platform alone": f"0 of {stl['platform_alone']['simulated_to_stall']}"})
    if rows:
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, width="stretch")

    pl = r.get("patient_loop", {})
    if pl:
        st.subheader("Patient loop: did the patient complete the care team's instruction?")
        c = st.columns(3)
        c[0].metric("Patient side completed the instruction", f"{pl['heartsafe_action_done']:.0%}",
                    f"{(pl['heartsafe_action_done'] - pl['platform_alone_action_done']) * 100:+.0f} pts vs one "
                    "message with no follow-up")
        c[1].metric("Confirmed by a caregiver or community health worker",
                    sum(v for k, v in pl["confirmed_by"].items() if k in ("caregiver", "community_health_worker")))
        c[2].metric("Reopened for the care team", pl["reopened"],
                    help="; ".join(f"{k}: {v}" for k, v in pl["reopen_reasons"].items()))
        lb = pd.DataFrame([{"remoteness": k, "arm": arm, "share": v[key]}
                           for k, v in pl["by_remoteness"].items()
                           for arm, key in (("HeartSafe", "heartsafe_action_done"),
                                            ("Platform alone", "platform_alone_action_done"))])
        st.altair_chart(alt.Chart(lb).mark_bar().encode(
            x=alt.X("remoteness:N", sort=["urban", "rural", "remote"], title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="arm:N", y=alt.Y("share:Q", title="Instruction completed", axis=alt.Axis(format="%")),
            color=alt.Color("arm:N", title=None)).properties(height=240), width="stretch")
        st.caption("Patient response times and no-response rates are simulation assumptions (config.yaml). "
                   "The simulation does not model the effect of treatment, so 'readings did not improve' "
                   "reopens are not a measure of care quality.")

    me = r.get("model_evidence", {})
    if me:
        st.subheader(f"Model evidence ({r['sources']['ecg']} for ECG, {r['sources']['mri']} for MRI)")
        rows = [{"Input": k.upper(), "Checks": v["n"], "Faulty or wrong": v["n_positive"],
                 "Caught (sensitivity)": f"{v['sensitivity']:.0%}" if v["sensitivity"] is not None else "-",
                 "Good results used (specificity)": f"{v['specificity']:.0%}" if v["specificity"] is not None else "-"}
                for k, v in me.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Ground truth = injected faults (ECG), or injected faults, LVEF error > 10 points or "
                   "LV Dice < 0.8 (MRI). Which patients get faulty recordings is a simulation assumption.")
    st.json(r, expanded=False)
    st.caption("Clinician response times, device fault rates and missing-data rates are simulation assumptions "
               "(config.yaml), identical for both arms.")

elif page == "Model reliability":
    st.header("Model reliability")
    st.caption("Held-out results of the ECG and cardiac MRI pipelines (run on SDRE). "
               "Copy fresh results with fetch_results.sh.")

    def jload(p):
        p = ROOT / p
        return json.load(open(p)) if p.exists() else None

    def img(p, caption):
        if (ROOT / p).exists():
            st.image(str(ROOT / p), caption=caption)

    tab_ecg, tab_mri = st.tabs(["ECG (PTB-XL lead I)", "Cardiac MRI (M&Ms)"])
    with tab_ecg:
        ev = jload("ecg/results/evaluation.json")
        if not ev:
            st.info("No ECG results on this machine yet.")
        else:
            p = ev["performance"]
            inj = ev["reliability"]["injected_faults"]["agent"]
            c = st.columns(4)
            c[0].metric("AUROC, PTB-XL test", f"{p['ptbxl_test_clean']['auroc_ci'][0]:.3f}")
            c[1].metric("AUROC, new device (AliveCor)", f"{p['cinc2017_device_shift']['auroc_ci'][0]:.3f}",
                        f"{-p['device_shift_auroc_drop']:+.3f}")
            c[2].metric("Injected faults caught", f"{inj['sensitivity']:.0%}")
            c[3].metric("Clean ECGs accepted", f"{inj['specificity']:.0%}")
            st.dataframe(pd.DataFrame([{"Fault": k, "Caught": f"{v:.0%}"} for k, v in
                                       ev["reliability"]["injected_faults"]["detection_by_fault"].items()]),
                         hide_index=True)
            a, b = st.columns(2)
            with a:
                img("ecg/results/accuracy_vs_coverage.png", "Accuracy rises as the least certain ECGs are set aside")
            with b:
                img("ecg/results/audit/forest_accuracy.png", "Subgroup accuracy, bootstrap 95% CIs")
    with tab_mri:
        rel = jload("imaging/results/reliability.json")
        if not rel:
            st.info("No imaging results on this machine yet.")
        else:
            cl = ROOT / "imaging/results/clean/exam_results.csv"
            if cl.exists():
                d = pd.read_csv(cl)
                v = d.groupby("vendor_name")[["dice_lv", "dice_myo", "dice_rv", "lvef_abs_err"]].mean().round(3)
                v.insert(0, "n", d.groupby("vendor_name").size())
                st.markdown("**Segmentation on 170 held-out subjects** (GE and Canon never seen in training)")
                st.dataframe(v.rename(columns={"lvef_abs_err": "LVEF error (points)"}), width="stretch")
            pol = []
            for f, name in (("reliability_v1_shift_escalates.json", "Scanner shift always escalates"),
                            ("reliability_v2_shift_caveat.json", "Shift = caveat"),
                            ("reliability.json", "Shift = caveat, severe shift escalates (current)")):
                x = jload(f"imaging/results/{f}")
                if x:
                    pol.append({"Policy": name,
                                "Faults caught": f"{x['agent_vs_injected_faults']['sensitivity']:.0%}",
                                "Clean exams flagged": f"{x['by_fault']['none']['flagged_share']:.0%}",
                                "Wrong predictions missed": x["missed_wrong_predictions_agent"]})
            pol.insert(0, {"Policy": "No agent", "Faults caught": "0%", "Clean exams flagged": "0%",
                           "Wrong predictions missed": rel["missed_wrong_predictions_baseline"]})
            st.markdown("**Reliability policy comparison**")
            st.dataframe(pd.DataFrame(pol), hide_index=True, width="stretch")
            img("imaging/results/audit/forest_lv_dice.png", "Subgroup LV Dice, bootstrap 95% CIs")

else:
    st.header("How HeartSafe works")
    st.markdown("""
0. **Who is enrolled:** adults hospitalised more than 48 h for heart failure, discharged in the last 14 days,
   NYHA II–IV, able to take readings (or with a caregiver). Excluded: dementia, long-term care, palliative care,
   transplant list or ventricular assist device, dialysis.
1. **The monitoring platform** (e.g. Medly, used in remote and Indigenous communities) collects daily weight, blood pressure,
   heart rate, oxygen saturation, symptoms and whether the diuretic was taken, plus post-discharge labs, and raises
   normal / caution / critical alerts.
2. **HeartSafe** checks each alert: is the reading plausible, was it a unit or cuff error, did the scale change
   (a nurse verifies it before the baseline changes), is the ECG good enough to trust, are MRI measurements reliable?
   Then it decides: accept, repeat measurement, or escalate.
3. Escalations become **tracked cases** with an explainable review. Deadlines trigger reminders and backup routing.
   After the clinician decides, the **patient loop** sends the instruction, waits for the patient to confirm and act,
   contacts a caregiver or community health worker if they do not, and checks the readings afterwards.
4. **Only a clinician** acknowledges, reviews and resolves; every action is in the audit log.
5. The **equity monitor** compares groups (remoteness, sex, age, device) and proposes safeguards that an administrator approves.

HeartSafe complements the monitoring platform and scheduled follow-up visits; it does not replace them.
""")
