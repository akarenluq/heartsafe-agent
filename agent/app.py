"""HeartSafe dashboard. Run from the agent/ folder:  streamlit run app.py"""
import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from heartsafe import DATA, load_config
from heartsafe.monitoring import SYMPTOM_TEXT
from heartsafe.orchestrator import orchestrate
from heartsafe.review import patient_evidence, summarize_ecg, summarize_mri
from heartsafe.tracker import PermissionDenied, Tracker
from heartsafe.watch import Watcher, followup_status

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="HeartSafe", page_icon=":anatomical_heart:", layout="wide")
if not (DATA / "heartsafe.db").exists():
    st.error("No results yet. In a terminal in the agent folder run:  python -m heartsafe.run")
    st.stop()

cfg = load_config()

# ---- look and feel ------------------------------------------------------------------------------
# Colours: validated categorical slots (HeartSafe = blue, platform alone = orange) and the reserved
# status colours for severity, always shown with a label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL, WARNING, GOOD = "#d03b3b", "#fab219", "#0ca30c"
INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARMS = alt.Scale(domain=["HeartSafe", "Platform alone"], range=[BLUE, ORANGE])

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1320px;}
h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
h3 {font-size: 1.15rem !important; margin-top: 0.6rem !important;}
.hs-sub {color: #52514e; font-size: 0.98rem; margin: 0.15rem 0 1.2rem;}
.hs-badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
  font-weight: 600; margin: 0 6px 4px 0; border: 1px solid rgba(11,11,11,0.10); white-space: nowrap;}
.hs-emergency {background: #7a0f0f; color: #fff;} .hs-critical {background: #d03b3b; color: #fff;} .hs-caution {background: #fab219; color: #0b0b0b;}
.hs-low, .hs-neutral {background: #f0efec; color: #0b0b0b;} .hs-good {background: #0ca30c; color: #fff;}
.hs-blue {background: #e6f0fb; color: #184f95; border-color: #cde2fb;}
.hs-label {color: #898781; font-size: 0.74rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; margin: 0.7rem 0 0.15rem;}
.hs-text {font-size: 0.95rem; line-height: 1.45;}
.hs-steps {display: flex; margin: 0.6rem 0 0.8rem;}
.hs-step {flex: 1; text-align: center; position: relative; font-size: 0.82rem; color: #52514e; padding: 0 4px;}
.hs-dot {width: 28px; height: 28px; border-radius: 50%; margin: 0 auto 6px; display: flex; align-items: center;
  justify-content: center; font-weight: 700; border: 2px solid #c3c2b7; background: #fcfcfb; color: #898781;
  position: relative; z-index: 1;}
.hs-step.done .hs-dot {background: #2a78d6; border-color: #2a78d6; color: #fff;}
.hs-step.bad .hs-dot {background: #d03b3b; border-color: #d03b3b; color: #fff;}
.hs-step:not(:last-child)::after {content: ""; position: absolute; top: 14px; left: calc(50% + 16px);
  right: calc(-50% + 16px); height: 2px; background: #e1e0d9;}
div[data-testid="stMetric"] {background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
  padding: 12px 16px;}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #52514e;}
/* sidebar: compact */
section[data-testid="stSidebar"] {width: 260px !important;}
[data-testid="stSidebarHeader"] {padding: 0.6rem 1rem 0 !important; height: auto !important;}
[data-testid="stSidebarNav"] {padding-top: 0.2rem !important;}
[data-testid="stSidebarNavItems"] {padding-bottom: 0.3rem !important;}
[data-testid="stSidebarNavLink"] {padding: 0.25rem 0.6rem !important; min-height: 0 !important;}
[data-testid="stSidebarNavLink"] span {font-size: 0.92rem !important;}
[data-testid="stSidebarNavSeparator"] {margin: 0.3rem 0 !important;}
[data-testid="stSidebarUserContent"] {padding: 0.4rem 1rem 1rem !important;}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {gap: 0.55rem !important;}
[data-testid="stSidebarUserContent"] label p {font-size: 0.8rem !important; color: #52514e;}
.hs-brand {font-size: 1.1rem; font-weight: 700; margin: 0;}
.hs-brand-sub {font-size: 0.8rem; color: #52514e; line-height: 1.35; margin: 0.1rem 0 0.9rem;}
.hs-clock {font-size: 0.76rem; color: #898781; line-height: 1.35; border-top: 1px solid #e1e0d9;
  padding-top: 0.5rem; margin-top: 0.2rem;}
</style>""", unsafe_allow_html=True)

SEV_LABEL = {"emergency": "🚨 Emergency", "critical": "🔴 Critical", "caution": "🟡 Caution", "low": "⚪ Low"}
SEV_ORDER = {"emergency": -1, "critical": 0, "caution": 1, "low": 2}
KIND_LABEL = {"clinical": "Clinical", "outreach": "Missed readings", "scale_check": "Scale check",
              "imaging": "MRI review", "patient_followup": "Patient follow-up"}
STATE_LABEL = {"created": "Waiting for acknowledgement", "acknowledged": "Acknowledged", "reviewed": "Reviewed",
               "resolved": "Resolved"}
DECISION = {  # what HeartSafe did with a platform alert: (plain words, group for the chart)
    "escalate": ("Escalated: new case for the nurse", "Escalated to the care team"),
    "escalate_update": ("Added to the open case", "Follow-up and patient loop"),
    "loop_update": ("Added to the case while the care plan runs", "Follow-up and patient loop"),
    "loop_closed": ("Patient loop closed: readings improved", "Follow-up and patient loop"),
    "loop_reopened": ("Reopened for the care team", "Escalated to the care team"),
    "outreach": ("Outreach: no readings for several days", "Follow-up and patient loop"),
    "repeat": ("Repeat reading requested (data issue)", "Data issue handled"),
    "filtered": ("Filtered: not supported after data correction", "Data issue handled"),
    "nurse_verify": ("Nurse asked to verify a possible new scale", "Data issue handled"),
    "nurse_verify_update": ("Held until the nurse verifies the scale", "Data issue handled"),
    "scale_verified": ("Nurse confirmed a new scale; re-baselined", "Data issue handled"),
    "imaging_review": ("MRI result sent for expert review", "Escalated to the care team"),
    "imaging_accept": ("MRI result accepted", "Follow-up and patient loop"),
    "emergency": ("EMERGENCY: chest pain or fainting with an abnormal ECG; video call within 15 min",
                  "Escalated to the care team"),
    "ecg_ok": ("ECG checked: no abnormality", "ECG checked"),
    "ecg_repeat": ("ECG not usable: patient asked to record it again", "Data issue handled"),
    "ecg_noted": ("Abnormal ECG with no alert: added to the chart", "ECG checked"),
}
GROUPS = alt.Scale(domain=["Escalated to the care team", "Data issue handled", "Follow-up and patient loop"],
                   range=[BLUE, ORANGE, AQUA])
AUDIT_LABEL = {"create": "Case created", "evidence": "Evidence added", "reminder": "Reminder sent",
               "acknowledge": "Acknowledged", "review": "Reviewed", "resolve": "Resolved",
               "severity_up": "Severity raised", "patient_notified": "Instruction sent to the patient",
               "patient_contact": "Helper contacted", "patient_confirm": "Instruction confirmed",
               "patient_action_done": "Action done", "loop_closed": "Loop closed: readings improved",
               "loop_reopened": "Reopened", "llm_review": "AI review written",
               "patient_message": "Message sent to the patient", "emergency_decision": "Video call decision"}
ISSUE_LABEL = {"ack_overdue": "Acknowledgement overdue", "resolve_overdue": "Resolution overdue",
               "patient_not_confirmed": "Patient has not confirmed the instruction"}


def badge(text, kind="neutral"):
    return f'<span class="hs-badge hs-{kind}">{text}</span>'


def label(text):
    st.markdown(f'<div class="hs-label">{text}</div>', unsafe_allow_html=True)


def text(t):
    st.markdown(f'<div class="hs-text">{t}</div>', unsafe_allow_html=True)


def header(title, sub):
    st.title(title)
    st.markdown(f'<div class="hs-sub">{sub}</div>', unsafe_allow_html=True)


def style(chart, height=260):
    return chart.properties(height=height).configure_view(stroke=None).configure_axis(
        gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=INK2, titleColor=INK2,
        labelFontSize=12, titleFontSize=12, titleFontWeight="normal").configure_legend(
        orient="top", title=None, labelColor=INK2, labelFontSize=12, symbolSize=90)


def day(h):
    return f"day {h / 24:.1f}"


def hours(x):
    return f"{x:.1f} h" if abs(x) < 48 else f"{x / 24:.1f} days"


# ---- data ---------------------------------------------------------------------------------------
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
EVERY_H = cfg["followup"]["watch_every_h"]


@st.cache_data(ttl=EVERY_H * 3600)
def watch_now(t):
    return Watcher(tr, cfg).tick(t)


# ---- sidebar ------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="hs-brand"><span style="color:#d03b3b">&#9829;</span> HeartSafe</p>'
                '<p class="hs-brand-sub">Safety and follow-up layer for remote heart-failure monitoring</p>',
                unsafe_allow_html=True)
    role = st.selectbox("Signed in as", ["clinician", "administrator"], format_func=str.capitalize)
    user = st.text_input("Name", "nurse_demo" if role == "clinician" else "admin_demo")
    st.markdown(f'<div class="hs-clock">Simulation clock: {day(now)}<br>Simulated data, modelled on Medly</div>',
                unsafe_allow_html=True)


# ---- one case -----------------------------------------------------------------------------------
def vitals_chart(pid, upto_day=None, window=21, measure="weight_kg"):
    titles = {"weight_kg": "Weight (kg)", "sbp": "Systolic BP (mmHg)", "hr": "Heart rate (bpm)"}
    f = feed[(feed.patient_id == pid) & feed.submitted]
    if upto_day is not None:
        f = f[(f.day <= upto_day) & (f.day > upto_day - window)]
    d = decisions[decisions.patient_id == pid].copy()
    d["action"] = d.decision.map(lambda x: DECISION.get(x, (x, ""))[0])
    d["group"] = d.decision.map(lambda x: DECISION.get(x, ("", "Follow-up and patient loop"))[1])
    d = d.merge(f[["day", measure]], on="day").drop_duplicates("day", keep="first")
    x = alt.X("day:Q", title="Day of monitoring")
    line = alt.Chart(f).mark_line(color=MUTED, strokeWidth=2).encode(
        x=x, y=alt.Y(f"{measure}:Q", title=titles[measure], scale=alt.Scale(zero=False)))
    pts = alt.Chart(f).mark_point(color=MUTED, size=20, filled=True).encode(
        x=x, y=f"{measure}:Q", tooltip=[alt.Tooltip("day:Q", title="Day"), alt.Tooltip(f"{measure}:Q",
                                                                                       title=titles[measure])])
    marks = alt.Chart(d).mark_point(size=130, filled=True, stroke="#fcfcfb", strokeWidth=2, opacity=1).encode(
        x=x, y=f"{measure}:Q", color=alt.Color("group:N", scale=GROUPS),
        tooltip=[alt.Tooltip("day:Q", title="Day"), alt.Tooltip("action:N", title="HeartSafe"),
                 alt.Tooltip("alert_rules:N", title="Platform alert")])
    return line + pts + marks


def loop_steps(loop):
    steps = [("Instruction sent", loop["notified_h"], False),
             ("Confirmed" + (f" by {loop['confirmed_by'].replace('_', ' ')}" if loop["confirmed_by"] else ""),
              loop["confirmed_h"], False),
             ("Action done", loop["action_done_h"], False),
             ({"closed": "Readings improved", "reopened": "Reopened"}.get(loop["state"], "Readings checked"),
              loop["closed_h"], loop["state"] == "reopened")]
    html = "".join(
        f'<div class="hs-step {"bad" if bad else "done" if t is not None else ""}"><div class="hs-dot">'
        f'{"!" if bad else "✓" if t is not None else i + 1}</div>{name}<br>'
        f'<span style="color:#898781">{day(t) if t is not None else "pending"}</span></div>'
        for i, (name, t, bad) in enumerate(steps))
    st.markdown(f'<div class="hs-steps">{html}</div>', unsafe_allow_html=True)


def audit_table(case_id):
    log = pd.DataFrame(tr.audit(case_id))

    def detail(d):
        if not isinstance(d, str) or not d:
            return ""
        d = json.loads(d)
        if "rules" in d:
            return "; ".join(d["rules"])
        return d.get("note") or d.get("outcome") or d.get("instruction") or (
            f"to {d['to'].replace('_', ' ')}" if "to" in d else ", ".join(f"{k}: {v}" for k, v in d.items()))
    return pd.DataFrame({"When": (log.t_h / 24).map(lambda x: f"day {x:.1f}"),
                         "Who": log.actor + " (" + log.role.str.replace("_", " ") + ")",
                         "What": log.action.map(lambda a: AUDIT_LABEL.get(a, a)),
                         "Details": log.detail.map(detail)})


def show_case(case):
    review = json.loads(case["review"])
    status = followup_status(case, tr, now)
    sev = case["severity"]
    st.markdown(badge(sev.capitalize(), sev) + badge(KIND_LABEL.get(case["kind"], case["kind"]), "blue")
                + badge(STATE_LABEL[status["state"]], "good" if status["state"] == "resolved" else "neutral")
                + "".join(badge(ISSUE_LABEL[i], "critical") for i in status["issues"])
                + f'<span style="color:#898781;font-size:0.85rem">Case {case["case_id"]} · {case["patient_id"]} · '
                  f'opened {day(case["created_h"])}</span>', unsafe_allow_html=True)

    if sev == "emergency":
        call = case["acknowledged_h"]
        st.error("**Emergency**: chest pain or fainting with a reliable abnormal ECG. The patient was told "
                 "\"If you feel worse, call 911.\"  \n"
                 + (f"Video call started after {60 * (call - case['created_h']):.0f} min.  \n" if call else
                    "Waiting for the clinician's video call (15-minute deadline).  \n")
                 + (f"Decision: {case['outcome']}" if case["outcome"] else
                    "Decision pending: the clinician records call 911 or cancel the alarm, with a reason. "
                    "HeartSafe never calls 911 itself."))
    t_review, t_follow, t_audit = st.tabs(["Review", "Follow-up", "Audit log"])
    with t_review:
        left, right = st.columns([3, 2], gap="large")
        with left:
            label("Why it was flagged")
            text(" · ".join(review["why_flagged"]) or "—")
            if review.get("symptoms"):
                label("Reported symptoms")
                text(", ".join(SYMPTOM_TEXT[s] for s in review["symptoms"]))
            ctx, en = review["context"], review.get("enrollment") or {}
            label("Patient")
            text(f"{ctx['age']} y, {ctx['sex']}, {ctx['remoteness']} community, {ctx['device'].replace('_', ' ')}"
                 + (f" · NYHA {en['nyha']}" if en.get("nyha") else "")
                 + (" · beta-blocker" if en.get("beta_blocker") else "")
                 + (" · has a caregiver" if en.get("caregiver") else ""))
            if review.get("trend"):
                w = [t["weight_kg"] for t in review["trend"]]
                label("Recent readings")
                text(f"Weight {w[0]:.1f} → {w[-1]:.1f} kg over the last {len(w)} readings · "
                     f"BP {review['trend'][-1]['sbp']:.0f} · HR {review['trend'][-1]['hr']:.0f}")
            if review.get("ecg"):
                label("ECG report")
                (st.success if review["ecg"]["reliable"] else st.warning)(summarize_ecg(review["ecg"]))
            if review.get("mri"):
                label("MRI report")
                (st.success if review["mri"]["reliable"] else st.warning)(
                    summarize_mri(review["mri"], cfg["review"].get("lvef_bands")))
            other = [c for c in review.get("caveats", []) if not c.startswith(("ECG not used", "MRI measurements"))
                     and c not in ((review.get("mri") or {}).get("caveats") or [])]
            if other:
                label("Data-quality notes")
                text("<br>".join(f"• {c}" for c in other))
            label("Suggested next step")
            st.info(review["next_step"])
            st.caption("HeartSafe does not make the clinical decision; the clinical team does.")
        with right:
            m = st.segmented_control("Measure", ["Weight", "Blood pressure", "Heart rate"], default="Weight",
                                     key=f"m{case['case_id']}", label_visibility="collapsed")
            col = {"Weight": "weight_kg", "Blood pressure": "sbp", "Heart rate": "hr"}[m or "Weight"]
            st.altair_chart(style(vitals_chart(case["patient_id"], review["day"], measure=col), 240),
                            width="stretch")
        if cfg["review"].get("llm_enabled") and st.button(
                "Explain with AI agent", key=f"llm{case['case_id']}",
                help="Optional: Claude calls read-only tools over this case and rewords the review."):
            with st.spinner("Agent gathering facts..."):
                out = orchestrate(case, tr, cfg["review"]["llm_model"], now_h=now)
            st.info(out["text"])
            st.caption(f"source: {out['source']}" + (f" · tools: {', '.join(out['tool_calls'])}"
                                                      if out["tool_calls"] else ""))
    with t_follow:
        label(f"Follow-up status (checked every {EVERY_H:g} h)")
        for line in status["lines"]:
            text("• " + line)
        loop = tr.loop(case["case_id"])
        if loop:
            label("Patient loop: the care team's instruction")
            text(loop["instruction"])
            loop_steps(loop)
            if loop["contacts"]:
                st.caption(f"Helpers contacted: {loop['contacts']} (caregiver or community health worker)")
            if loop["state"] == "reopened":
                st.warning(f"Reopened: {loop['outcome']}")
    with t_audit:
        st.dataframe(audit_table(case["case_id"]), hide_index=True, width="stretch")
        st.caption("Append-only. Only a clinician can acknowledge, review or resolve; only the patient side can confirm.")

    nxt = {"created": "acknowledge", "acknowledged": "review", "reviewed": "resolve"}.get(case["state"])
    if nxt and role != "clinician":
        st.caption(f"🔒 Next step: {nxt}. Only a clinician can act on this case; switch to Clinician in the sidebar.")
    elif nxt:
        with st.container(border=True):
            note = st.text_area("Resolution note (required)", key=f"note{case['case_id']}") if nxt == "resolve" else ""
            if st.button(f"{nxt.capitalize()} case {case['case_id']}", type="primary", key=f"act{case['case_id']}"):
                try:
                    tr.act(case["case_id"], nxt, now, user, role, note)
                    tr.commit()
                    st.success(f"Case {case['case_id']}: {nxt} recorded by {user}.")
                    st.rerun()
                except PermissionDenied as e:
                    st.error(f"Not allowed: {e}. HeartSafe never closes a case on its own.")
                except ValueError as e:
                    st.error(str(e))


# ---- pages --------------------------------------------------------------------------------------
def page_queue():
    header("Clinician queue", f"Open cases at simulation {day(now)}, most urgent first. Select a row to open the case.")
    cases = pd.DataFrame(tr.cases())
    open_ = cases[cases.state != "resolved"].copy()
    open_["overdue"] = (open_.state == "created") & (open_.ack_deadline_h < now)
    found = watch_now(now)
    c = st.columns(4)
    c[0].metric("Open cases", len(open_))
    c[1].metric("Emergency / critical", f"{int((open_.severity == 'emergency').sum())} / "
                                        f"{int((open_.severity == 'critical').sum())}")
    c[2].metric("Acknowledgement overdue", int(open_.overdue.sum()))
    c[3].metric("Need a follow-up action", len(found), help=f"From the follow-up watch, every {EVERY_H:g} h")

    @st.fragment(run_every=EVERY_H * 3600)
    def followup_watch():
        items = watch_now(now)
        with st.container(border=True):
            st.markdown(f"**Follow-up watch** · checks every case and patient loop every {EVERY_H:g} h · "
                        f"last check {day(now)}")
            if not items:
                st.success("Nothing is overdue.")
            else:
                # only the lines that explain the issue, so the row stays readable
                key = ("deadline passed", "reminder", "Helpers contacted", "not confirmed")
                st.dataframe(pd.DataFrame([{
                    "Case": s["case_id"], "Patient": s["patient_id"], "Severity": SEV_LABEL[s["severity"]],
                    "Issue": ", ".join(ISSUE_LABEL[i] for i in s["issues"]),
                    "Details": " ".join(x.replace("; the confirmation deadline passed", ", deadline passed")
                                        for x in s["lines"] if any(k in x for k in key))}
                    for s in items]), hide_index=True, width="stretch",
                    column_config={"Case": st.column_config.NumberColumn(width="small"),
                                   "Severity": st.column_config.TextColumn(width="small"),
                                   "Details": st.column_config.TextColumn(width="large",
                                                                          help="Select the case below for the full status")})
    followup_watch()

    f1, f2 = st.columns(2)
    sev = f1.multiselect("Severity", ["emergency", "critical", "caution", "low"], format_func=lambda s: SEV_LABEL[s])
    kind = f2.multiselect("Type", sorted(open_.kind.unique()), format_func=lambda k: KIND_LABEL.get(k, k))
    view = open_[open_.severity.isin(sev or SEV_ORDER) & open_.kind.isin(kind or open_.kind.unique())]
    view = view.assign(rank=view.severity.map(SEV_ORDER)).sort_values(["overdue", "rank", "created_h"],
                                                                      ascending=[False, True, True])
    due = view.ack_deadline_h - now
    table = pd.DataFrame({
        "Case": view.case_id, "Patient": view.patient_id, "Severity": view.severity.map(SEV_LABEL),
        "Type": view.kind.map(lambda k: KIND_LABEL.get(k, k)), "State": view.state.map(STATE_LABEL),
        "Acknowledge by": [("overdue " + hours(-x)) if (s == "created" and x < 0) else
                           (f"in {hours(x)}" if s == "created" else "—") for x, s in zip(due, view.state)],
        "Opened": (view.created_h / 24).map(lambda x: f"day {x:.1f}"), "Reminders": view.reminders,
        "Assigned to": view.assigned_to.str.capitalize() + " nurse"})
    if not len(table):
        st.info("No open cases match the filters.")
        return
    sel = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                       height=min(36 * len(table) + 40, 380))
    rows = sel.selection.rows
    cid = int(table.iloc[rows[0]].Case) if rows else int(table.iloc[0].Case)
    st.divider()
    show_case(tr.get(cid))


def page_patient():
    header("Patient journey", "Every platform alert for one patient, what HeartSafe did with it, and each case.")
    ids = ["DEMO-MARY"] + sorted(p for p in decisions.patient_id.unique() if p != "DEMO-MARY")
    pid = st.selectbox("Patient", ids, format_func=lambda p: "DEMO-MARY · demo patient" if p == "DEMO-MARY" else p)
    p = patients[patients.patient_id == pid].iloc[0]
    d = decisions[decisions.patient_id == pid]
    pcases = [c for c in tr.cases() if c["patient_id"] == pid]
    c = st.columns(5)
    c[0].metric("Patient", f"{p.age} y, {p.sex}")
    c[1].metric("Community", str(p.remoteness).capitalize())
    c[2].metric("NYHA class", p.nyha if "nyha" in p else "—")
    c[3].metric("Platform alerts", int((d.alert_level > 0).sum()))
    c[4].metric("Cases opened", len(pcases))
    st.caption(f"{p.device.replace('_', ' ').capitalize()} · ECG device: {'yes' if p.ecg_kit else 'no'} · "
               f"caregiver: {'yes' if p.caregiver else 'no'} · discharged {p.days_since_discharge} days before "
               "enrollment")

    with st.container(border=True):
        m = st.segmented_control("Measure", ["Weight", "Blood pressure", "Heart rate"], default="Weight",
                                 key="pj_measure")
        col = {"Weight": "weight_kg", "Blood pressure": "sbp", "Heart rate": "hr"}[m or "Weight"]
        st.altair_chart(style(vitals_chart(pid, measure=col), 280), width="stretch")
        st.caption("Gray line: daily readings. Coloured dots: days HeartSafe acted on a platform alert "
                   "(hover for details).")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("ECG and MRI reports")
        ev = patient_evidence(d, [json.loads(c["review"]) for c in pcases], cfg["review"].get("lvef_bands"))
        if ev:
            for line in ev:
                text("• " + line)
        else:
            st.caption("No ECG or MRI for this patient.")
    with right:
        st.subheader("Alerts and HeartSafe actions")
        acted = d[d.decision.notna()]
        st.dataframe(pd.DataFrame({"Day": acted.day, "Platform alert": acted.alert_rules.fillna("—"),
                                   "HeartSafe": acted.decision.map(lambda x: DECISION.get(x, (x,))[0])}),
                     hide_index=True, width="stretch", height=min(36 * len(acted) + 40, 320))

    st.subheader(f"Cases ({len(pcases)})")
    for case in pcases:
        icon = {"emergency": "🚨", "critical": "🔴", "caution": "🟡", "low": "⚪"}[case["severity"]]
        with st.expander(f"{icon} Case {case['case_id']} · {KIND_LABEL.get(case['kind'], case['kind'])} · "
                         f"{case['severity']} · {day(case['created_h'])} · {STATE_LABEL[case['state']].lower()}"):
            show_case(case)


def page_equity():
    header("Equity audit", "Are some groups served worse? Observable measures only, with bootstrap 95% confidence "
                           "intervals. A gap is flagged only when a group's whole interval is worse than overall.")
    eq = load("equity.csv")
    t_gap, t_safe = st.tabs(["Subgroup gaps", "Safeguards"])
    with t_gap:
        gaps = eq[eq.gap.astype(bool)]
        if len(gaps):
            st.markdown("**Gaps found:** " + " ".join(
                badge(f"{r.label.split(' (')[0]}: {r.group} = {r.value}", "critical") for r in gaps.itertuples()),
                unsafe_allow_html=True)
        metric = st.selectbox("Measure", eq.metric.unique(), format_func=lambda m: eq[eq.metric == m].label.iloc[0])
        e = eq[eq.metric == metric].copy()
        e["row"] = e.group.replace({"overall": "Overall", "age_band": "age"}) + " · " + e.value.astype(str)
        e["status"] = e.gap.map({True: "Gap", False: "No gap"})
        overall = e[e.group == "overall"]["mean"].iloc[0]
        status = alt.Scale(domain=["Gap", "No gap"], range=[CRITICAL, INK2])
        base = alt.Chart(e).encode(y=alt.Y("row:N", sort=None, title=None))
        tip = ["row", "n", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("ci_low:Q", format=".3f"),
               alt.Tooltip("ci_high:Q", format=".3f"), "status"]
        chart = (base.mark_rule(strokeWidth=2).encode(x=alt.X("ci_low:Q", title=e.label.iloc[0]), x2="ci_high:Q",
                                                     color=alt.Color("status:N", scale=status), tooltip=tip)
                 + base.mark_point(filled=True, size=90).encode(x="mean:Q", color=alt.Color("status:N", scale=status),
                                                                tooltip=tip)
                 + base.transform_filter("datum.gap").mark_text(align="left", dx=8, color=CRITICAL, fontWeight="bold")
                 .encode(x="ci_high:Q", text=alt.value("gap"))
                 + alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x="x:Q"))
        st.altair_chart(style(chart, 28 * len(e) + 40), width="stretch")
        st.caption("Dot = group average, line = 95% CI, dashed line = overall average. Groups under 30 are never "
                   "flagged.")
    with t_safe:
        props = load("proposals.json") or []
        sg_path = DATA / "safeguards.json"
        approved = json.load(open(sg_path)) if sg_path.exists() else []
        approved_ids = {s["id"] for s in approved if s.get("approved")}
        names = {"tighter_ack_deadline": "Tighter acknowledgement deadline",
                 "earlier_patient_followup": "Contact helpers earlier", "recommendation": "Program recommendation"}
        st.markdown("#### Safeguards that change the agent")
        st.caption("Each one needs administrator approval. Its evidence and its workload cost are shown.")
        cols = st.columns(2)
        for i, p in enumerate([p for p in props if p["type"] != "recommendation"]):
            with cols[i % 2].container(border=True):
                st.markdown(f"**{names.get(p['type'], p['type'])}** · {p['field']} = {p['value']}")
                st.caption(p["evidence"])
                text(f"Effect: {p['effect']}")
                if p["id"] in approved_ids:
                    who = next(s.get("approved_by", "?") for s in approved if s["id"] == p["id"])
                    st.success(f"Approved by {who}")
                elif role == "administrator":
                    if st.button("Approve safeguard", key=p["id"], type="primary"):
                        approved.append({**p, "approved": True, "approved_by": user})
                        json.dump(approved, open(sg_path, "w"), indent=2)
                        st.rerun()
                else:
                    st.caption("🔒 Sign in as administrator to approve.")
        recs = [p for p in props if p["type"] == "recommendation"]
        if recs:
            st.markdown("#### Program recommendations (not automated)")
            for p in recs:
                text(f"• **{p['field']} = {p['value']}**: {p['effect'].split(' (')[0]}. "
                     f"<span style='color:#898781'>{p['evidence']}</span>")
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
            st.markdown("#### What the approved safeguards changed")
            c = st.columns(3)
            c[0].metric("Extra clinical cases / 100 patient-months", f"{effect['extra_clinical_cases_per_100_pm']:+.1f}")
            c[1].metric("Extra false alarms / 100 patient-months", f"{effect['extra_false_alarms_per_100_pm']:+.1f}")
            c[2].metric("Extra reminders", f"{effect['extra_reminders']:+d}")
            st.dataframe(pd.DataFrame(effect["targeted_groups"]), hide_index=True, width="stretch")
            st.caption("Before = the last run with no approved safeguards. Same cohort and seed.")


def bars(df, y, title, fmt=None):
    return style(alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke="#fcfcfb",
                                        strokeWidth=2).encode(
        x=alt.X("remoteness:N", sort=["urban", "rural", "remote"], title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("arm:N", sort=["HeartSafe", "Platform alone"]),
        y=alt.Y(f"{y}:Q", title=title, axis=alt.Axis(format=fmt) if fmt else alt.Axis()),
        color=alt.Color("arm:N", scale=ARMS),
        tooltip=["remoteness", "arm", alt.Tooltip(f"{y}:Q", format=fmt or ".1f")]), 260)


def page_results():
    header("Results: HeartSafe vs the platform alone",
           "Same simulated patients and nurses in both arms (2,001 patients × 90 days); the only difference is "
           "whether HeartSafe sits between the platform's alerts and the care team.")
    r = load("evaluation.json")
    te, pl, q = r["true_episodes"], r.get("patient_loop", {}), r["nurse_queue_per_100_patient_months"]
    c = st.columns(4)
    h, b = te["median_hours_first_alert_to_ack"]["heartsafe"], te["median_hours_first_alert_to_ack"]["platform_alone"]
    c[0].metric("Median hours to acknowledge", f"{h}", f"{h - b:+.1f} h vs platform alone", delta_color="inverse",
                help="True heart-failure episodes, first alert to acknowledgement")
    c[1].metric("Faulty-reading alerts stopped", f"{r['fault_driven_alerts']['stopped_by_heartsafe']:.0%}",
                help=f"{r['fault_driven_alerts']['n']} alerts caused by device or entry errors; the platform alone "
                     "forwards all of them")
    c[2].metric("False alarms / 100 patient-months", f"{q['false_alarms_heartsafe']:.0f}",
                f"{q['false_alarms_heartsafe'] - q['false_alarms_platform_alone']:+.0f} vs platform alone",
                delta_color="inverse")
    if pl:
        c[3].metric("Patients who completed the instruction", f"{pl['heartsafe_action_done']:.0%}",
                    f"{(pl['heartsafe_action_done'] - pl['platform_alone_action_done']) * 100:+.0f} pts",
                    help="Compared with one message and no follow-up")

    t1, t2, t3, t4 = st.tabs(["Speed and follow-up", "Patient loop", "Model evidence", "Assumptions and raw data"])
    with t1:
        a, b2 = st.columns([3, 2], gap="large")
        with a:
            by = pd.DataFrame([{"remoteness": k, "arm": arm, "hours": v[f"{key}_mean"]}
                               for k, v in te["hours_to_ack_by_remoteness"].items()
                               for arm, key in (("HeartSafe", "heartsafe"), ("Platform alone", "platform_alone"))])
            st.markdown("**Mean hours from first alert to acknowledgement, true episodes**")
            st.altair_chart(bars(by, "hours", "Hours"), width="stretch")
        with b2:
            res, stl = r["resolution"], r["stalled_cases"]
            st.markdown("**Follow-up**")
            st.dataframe(pd.DataFrame([
                {"Measure": "Acknowledged within 24 h (true episodes)",
                 "HeartSafe": f"{te['acknowledged_within_24h']['heartsafe']:.0%}",
                 "Platform alone": f"{te['acknowledged_within_24h']['platform_alone']:.0%}"},
                {"Measure": "Resolved within deadline", "HeartSafe": f"{res['heartsafe']['resolved_within_deadline']:.0%}",
                 "Platform alone": f"{res['platform_alone']['resolved_within_deadline']:.0%}"},
                {"Measure": "Median hours to resolution", "HeartSafe": str(res["heartsafe"]["median_hours_to_resolution"]),
                 "Platform alone": str(res["platform_alone"]["median_hours_to_resolution"])},
                {"Measure": "Stalled cases found",
                 "HeartSafe": f"{stl['heartsafe']['flagged_by_reminder']} of {stl['heartsafe']['simulated_to_stall']}",
                 "Platform alone": f"0 of {stl['platform_alone']['simulated_to_stall']}"},
                {"Measure": "Nurse queue items / 100 patient-months",
                 "HeartSafe": f"{q['heartsafe_clinical_cases'] + q['heartsafe_nurse_verifications'] + q['heartsafe_patient_followups']:.0f}",
                 "Platform alone": f"{q['platform_alone_alert_items']:.0f}"}]), hide_index=True, width="stretch")
    with t2:
        if pl:
            a, b2 = st.columns([3, 2], gap="large")
            with a:
                lb = pd.DataFrame([{"remoteness": k, "arm": arm, "share": v[key]}
                                   for k, v in pl["by_remoteness"].items()
                                   for arm, key in (("HeartSafe", "heartsafe_action_done"),
                                                    ("Platform alone", "platform_alone_action_done"))])
                st.markdown("**Share of instructions the patient side completed**")
                st.altair_chart(bars(lb, "share", "Completed", ".0%"), width="stretch")
            with b2:
                st.metric("Confirmed by a caregiver or community health worker",
                          sum(v for k, v in pl["confirmed_by"].items() if k in ("caregiver", "community_health_worker")))
                st.metric("Median hours from instruction to action", pl["median_hours_to_action"])
                st.metric("Reopened for the care team", pl["reopened"],
                          help="; ".join(f"{k}: {v}" for k, v in pl.get("reopen_reasons", {}).items()))
    with t3:
        me = r.get("model_evidence", {})
        if me:
            st.dataframe(pd.DataFrame([
                {"Input": k.upper(), "Results checked": v["n"], "Faulty or wrong": v["n_positive"],
                 "Caught (sensitivity)": f"{v['sensitivity']:.0%}" if v["sensitivity"] is not None else "—",
                 "Good results used (specificity)": f"{v['specificity']:.0%}" if v["specificity"] is not None else "—"}
                for k, v in me.items()]), hide_index=True, width="stretch")
            st.caption(f"Sources: {r['sources']['ecg']} (ECG), {r['sources']['mri']} (MRI). Ground truth = injected "
                       "faults (ECG), or injected faults, LVEF error > 10 points or LV Dice < 0.8 (MRI).")
    with t4:
        scr = load("screening.json")
        if scr:
            text(f"<b>Enrollment:</b> {scr['enrolled']:,} of {scr['candidates']:,} screened candidates met the inclusion "
                 "criteria. Most common reasons not eligible: "
                 + "; ".join(f"{k} ({v})" for k, v in list(scr["reasons"].items())[:4]) + ".")
        st.info("Clinician and patient response times, device fault rates and missing-data rates are simulation "
                "assumptions (config.yaml), identical for both arms. The simulation does not model the effect of "
                "treatment.")
        with st.expander("Raw evaluation results (JSON)"):
            st.json(r, expanded=False)


def page_models():
    header("Model reliability", "Held-out results of the ECG and cardiac MRI pipelines (trained and run on SDRE).")

    def jload(p):
        p = ROOT / p
        return json.load(open(p)) if p.exists() else None

    def img(p, caption):
        if (ROOT / p).exists():
            st.image(str(ROOT / p), caption=caption)

    tab_ecg, tab_mri = st.tabs(["ECG · PTB-XL lead I", "Cardiac MRI · M&Ms"])
    with tab_ecg:
        ev = jload("ecg/results/evaluation.json")
        if not ev:
            st.info("No ECG results on this machine yet (copy them with fetch_results.sh).")
        else:
            p = ev["performance"]
            inj = ev["reliability"]["injected_faults"]["agent"]
            c = st.columns(4)
            c[0].metric("AUROC, PTB-XL test", f"{p['ptbxl_test_clean']['auroc_ci'][0]:.3f}")
            c[1].metric("AUROC, new device (AliveCor)", f"{p['cinc2017_device_shift']['auroc_ci'][0]:.3f}",
                        f"{-p['device_shift_auroc_drop']:+.3f}")
            c[2].metric("Injected faults caught", f"{inj['sensitivity']:.0%}")
            c[3].metric("Clean ECGs accepted", f"{inj['specificity']:.0%}")
            a, b = st.columns([1, 2], gap="large")
            with a:
                st.markdown("**Faults caught, by type**")
                st.dataframe(pd.DataFrame([{"Fault": k.replace("_", " "), "Caught": f"{v:.0%}"} for k, v in
                                           ev["reliability"]["injected_faults"]["detection_by_fault"].items()]),
                             hide_index=True, width="stretch")
            with b:
                img("ecg/results/accuracy_vs_coverage.png", "Accuracy rises as the least certain ECGs are set aside")
            img("ecg/results/audit/forest_accuracy.png", "Subgroup accuracy, bootstrap 95% CIs")
    with tab_mri:
        rel = jload("imaging/results/reliability.json")
        if not rel:
            st.info("No imaging results on this machine yet (copy them with fetch_results.sh).")
        else:
            cl = ROOT / "imaging/results/clean/exam_results.csv"
            if cl.exists():
                d = pd.read_csv(cl)
                v = d.groupby("vendor_name")[["dice_lv", "dice_myo", "dice_rv", "lvef_abs_err"]].mean().round(3)
                v.insert(0, "n", d.groupby("vendor_name").size())
                st.markdown("**Segmentation on 170 held-out subjects** (GE and Canon never seen in training)")
                st.dataframe(v.rename(columns={"dice_lv": "Dice LV", "dice_myo": "Dice myocardium", "dice_rv": "Dice RV",
                                               "lvef_abs_err": "LVEF error (points)"}), width="stretch")
            pol = [{"Policy": "No agent", "Faults caught": "0%", "Clean exams flagged": "0%",
                    "Wrong predictions missed": rel["missed_wrong_predictions_baseline"]}]
            for f, name in (("reliability_v1_shift_escalates.json", "Scanner shift always escalates"),
                            ("reliability_v2_shift_caveat.json", "Shift = caveat"),
                            ("reliability.json", "Shift = caveat, severe shift escalates (used)")):
                x = jload(f"imaging/results/{f}")
                if x:
                    pol.append({"Policy": name, "Faults caught": f"{x['agent_vs_injected_faults']['sensitivity']:.0%}",
                                "Clean exams flagged": f"{x['by_fault']['none']['flagged_share']:.0%}",
                                "Wrong predictions missed": x["missed_wrong_predictions_agent"]})
            st.markdown("**Reliability policy comparison**")
            st.dataframe(pd.DataFrame(pol), hide_index=True, width="stretch")
            img("imaging/results/audit/forest_lv_dice.png", "Subgroup LV Dice, bootstrap 95% CIs")


def page_how():
    header("How HeartSafe works", "A trust and coordination layer on top of the monitoring platform. It complements "
                                  "the platform and scheduled follow-up visits; it does not replace them.")
    steps = [
        ("1 · Enrollment", "Adults hospitalised 48 h or more for heart failure, discharged in the last 14 days, NYHA "
         "II–IV, able to take readings or with a caregiver. Excluded: dementia, long-term care, palliative care, "
         "transplant list or LVAD, dialysis. Personal BP / heart-rate limits can be set."),
        ("2 · The platform", "Collects daily weight, blood pressure, heart rate, oxygen saturation, symptoms and "
         "diuretic use, plus post-discharge labs, and raises normal / caution / critical alerts."),
        ("3 · Reliability checks", "Is the reading plausible? A unit or cuff error? A new scale (a nurse verifies it)? "
         "Is the ECG good enough to trust? Are the MRI measurements reliable? Then: accept, repeat, or escalate."),
        ("4 · Tracked cases", "Each escalation becomes a case with an explainable review. Deadlines trigger reminders "
         "and backup routing; a follow-up watch re-checks every case every few hours."),
        ("5 · Patient loop", "After the clinician decides, the instruction goes to the patient; a caregiver or "
         "community health worker is contacted if they do not confirm. Readings are checked after the action."),
        ("6 · Equity monitor", "Compares groups (remoteness, sex, age, device) and proposes safeguards, each with its "
         "cost, that an administrator approves."),
    ]
    for row in (steps[:3], steps[3:]):
        cols = st.columns(3)
        for col, (t, body) in zip(cols, row):
            with col.container(border=True, height=230):
                st.markdown(f"**{t}**")
                text(body)
    st.info("Only a clinician acknowledges, reviews and resolves a case, and only the patient side confirms an "
            "instruction. Every action is in the audit log.")


nav = st.navigation([
    st.Page(page_queue, title="Clinician queue", icon=":material/inbox:", default=True),
    st.Page(page_patient, title="Patient journey", icon=":material/person:"),
    st.Page(page_equity, title="Equity audit", icon=":material/balance:"),
    st.Page(page_results, title="Results", icon=":material/monitoring:"),
    st.Page(page_models, title="Model reliability", icon=":material/verified:"),
    st.Page(page_how, title="How it works", icon=":material/help:"),
])
nav.run()
