"""HeartSafe dashboard. Run from the agent/ folder:  streamlit run app.py"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from heartsafe import DATA, load_config
from heartsafe.review import llm_rewrite, render
from heartsafe.tracker import PermissionDenied, Tracker

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
st.sidebar.caption("Trust and coordination layer on top of Medly-style remote monitoring")
role = st.sidebar.selectbox("Signed in as", ["clinician", "administrator"])
user = st.sidebar.text_input("Name", "nurse_demo" if role == "clinician" else "admin_demo")
page = st.sidebar.radio("View", ["Patient journey", "Clinician queue", "Equity audit", "Results vs Medly alone",
                                 "How it works"])
st.sidebar.caption(f"Simulation clock: day {now / 24:.0f}. Medly data here is simulated; Medly is not open source.")


def vitals_chart(pid, upto_day=None, window=None):
    f = feed[(feed.patient_id == pid) & feed.submitted]
    if upto_day is not None:
        f = f[(f.day <= upto_day) & (f.day > upto_day - (window or 21))]
    d = decisions[decisions.patient_id == pid]
    base = alt.Chart(f).mark_line(point=True).encode(x=alt.X("day:Q", title="Day"),
                                                     y=alt.Y("weight_kg:Q", title="Weight (kg)", scale=alt.Scale(zero=False)))
    marks = alt.Chart(d.merge(f[["day", "weight_kg"]], on="day")).mark_point(size=160, filled=True).encode(
        x="day:Q", y="weight_kg:Q", color=alt.Color("decision:N", title="HeartSafe action"),
        tooltip=["day", "decision", "severity", "medly_rules", "flags"])
    return (base + marks).properties(height=260)


def show_case(case):
    review = json.loads(case["review"])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(render(review))
        if st.button("Reword with LLM", key=f"llm{case['case_id']}"):
            text, src = llm_rewrite(review, cfg["review"]["llm_model"])
            st.info(text)
            st.caption(f"source: {src}")
    with c2:
        st.altair_chart(vitals_chart(case["patient_id"], review["day"]), width="stretch")
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
    st.caption(f"{p.age} y, {p.sex}, {p.remoteness} community, {p.device.replace('_', ' ')}, "
               f"ECG device: {'yes' if p.ecg_kit else 'no'}")
    st.altair_chart(vitals_chart(pid), width="stretch")
    d = decisions[decisions.patient_id == pid][["day", "medly_rules", "decision", "severity", "flags"]]
    st.markdown("**Every Medly alert and what HeartSafe did with it**")
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
                st.success("Approved. Re-run `python -m heartsafe.run` to apply it.")
            elif role == "administrator":
                if st.button("Approve safeguard", key=p["id"]):
                    approved.append({**p, "approved": True, "approved_by": user})
                    json.dump(approved, open(sg_path, "w"), indent=2)
                    st.rerun()
            else:
                st.caption("Needs administrator approval.")

elif page == "Results vs Medly alone":
    st.header("HeartSafe vs Medly alone (simulated cohort)")
    r = load("evaluation.json")
    te = r["true_episodes"]
    c = st.columns(4)
    c[0].metric("Faulty-reading alerts stopped", f"{r['fault_driven_alerts']['stopped_by_heartsafe']:.0%}",
                help=f"{r['fault_driven_alerts']['n']} alerts caused by device or entry errors; Medly alone forwards all")
    c[1].metric("Mean hours, first alert to acknowledgement",
                f"{te['mean_hours_first_alert_to_ack']['heartsafe']}",
                f"{te['mean_hours_first_alert_to_ack']['heartsafe'] - te['mean_hours_first_alert_to_ack']['medly_alone']:+.1f} vs Medly alone",
                delta_color="inverse")
    c[2].metric("True episodes acknowledged within 24 h", f"{te['acknowledged_within_24h']['heartsafe']:.0%}",
                f"{(te['acknowledged_within_24h']['heartsafe'] - te['acknowledged_within_24h']['medly_alone']) * 100:+.0f} pts")
    c[3].metric("Alert items never acknowledged (Medly alone)", r["followup"]["medly_alone_alert_items_never_acknowledged"],
                help="With HeartSafe every case is tracked to acknowledgement (reminders, backup routing)")
    by = pd.DataFrame([{"remoteness": k, "arm": arm, "hours": v[f"{key}_mean"]}
                       for k, v in te["hours_to_ack_by_remoteness"].items()
                       for arm, key in (("HeartSafe", "heartsafe"), ("Medly alone", "medly_alone"))])
    st.altair_chart(alt.Chart(by).mark_bar().encode(
        x=alt.X("remoteness:N", sort=["urban", "rural", "remote"], title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="arm:N", y=alt.Y("hours:Q", title="Mean hours to acknowledge"),
        color=alt.Color("arm:N", title=None)).properties(height=280), width="stretch")
    st.json(r, expanded=False)
    st.caption("Clinician response times, device fault rates and missing-data rates are simulation assumptions "
               "(config.yaml), identical for both arms.")

else:
    st.header("How HeartSafe works")
    st.markdown("""
1. **Medly** collects daily weight, blood pressure, heart rate and symptoms and raises normal / caution / critical alerts.
2. **HeartSafe** checks each alert: is the reading plausible, was it a unit or cuff error, did the scale change,
   is the ECG good enough to trust, are MRI measurements reliable? Then it decides: accept, repeat measurement, or escalate.
3. Escalations become **tracked cases** with an explainable review. Deadlines trigger reminders and backup routing.
4. **Only a clinician** acknowledges, reviews and resolves; every action is in the audit log.
5. The **equity monitor** compares groups (remoteness, sex, age, device) and proposes safeguards that an administrator approves.

HeartSafe complements Medly and scheduled follow-up visits; it does not replace them.
""")
