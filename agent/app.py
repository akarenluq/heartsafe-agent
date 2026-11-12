"""HeartSafe dashboard. Run from the agent/ folder:  streamlit run app.py"""
import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from heartsafe import DATA, load_config
from heartsafe.agent import checkin_from_sleep_log
from heartsafe.monitoring import SYMPTOM_TEXT
from heartsafe.orchestrator import orchestrate
from heartsafe.review import patient_evidence, summarize_ecg, summarize_mri
from heartsafe.selfcare import KEPT_AWAKE, SelfCareLog, breathing_status, sleep_status, with_logged
from heartsafe.selfcare import describe as describe_selfcare
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
              "imaging": "MRI review", "patient_followup": "Patient follow-up", "sleep_check": "Sleep check-in"}
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
    "ecg_new_af": ("New atrial fibrillation on the ECG: reported to the care team", "Escalated to the care team"),
    "ecg_known_af": ("ECG checked: known atrial fibrillation, not a new finding", "ECG checked"),
    "sleep_check": ("Sleep worse than usual: check-in call for the nurse", "Follow-up and patient loop"),
    "sleep_noted": ("Sleep worse than usual: added to the open case", "Follow-up and patient loop"),
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


@st.cache_resource
def selfcare_log():
    return SelfCareLog(DATA / "selfcare.db")  # its own file: a cohort re-run does not erase it


tr, sc_log = tracker(), selfcare_log()
feed, patients, decisions = load("feed.csv"), load("patients.csv"), load("decisions.csv")
now = load("run_info.json")["horizon_h"]
TODAY = int(now // 24)  # the morning the patient logs last night's sleep
EVERY_H = cfg["followup"]["watch_every_h"]
MEASURES = {"Weight": "weight_kg", "Blood pressure": "sbp", "Heart rate": "hr",
            **({"Sleep": "sleep_hours"} if "sleep_hours" in feed.columns else {})}
SELFCARE_COLS = [c for c in ("sleep_hours", "sleep_awakenings", "breathing_sessions") if c in feed.columns]


def selfcare_history(pid, upto_day=None):
    """The patient's last 14 days of platform readings with the entries logged in the app merged
    in, oldest first."""
    f = feed[(feed.patient_id == pid) & feed.submitted]
    if upto_day is not None:
        f = f[f.day <= upto_day]
    hist = with_logged(f[["day", "weight_kg", "sbp", "hr"] + SELFCARE_COLS].tail(14).to_dict("records"),
                       sc_log.days(pid))
    return [h for h in hist if upto_day is None or h["day"] <= upto_day]


def logged_table(pid):
    """What the patient side logged in the app, newest first, in plain words."""
    rows = []
    for e in reversed(sc_log.entries(pid)):
        x = e["detail"]
        if e["kind"] == "sleep":
            what = (f"Slept {x['hours']:g} h, woke {x['awakenings']} time{'' if x['awakenings'] == 1 else 's'}"
                    + ("; slept propped up (short of breath lying flat)" if x["propped_up"] else "")
                    + ("; woke up short of breath" if x["woke_breathless"] else "")
                    + (f"; kept awake by: {', '.join(x['kept_awake'])}" if x["kept_awake"] else ""))
        else:
            what = f"{x['exercise']}, {x['minutes']:g} min"
        rows.append({"Day": e["day"], "Entry": "Sleep" if e["kind"] == "sleep" else "Breathing exercise",
                     "Details": what, "Logged by": f"{e['actor']} ({e['role'].replace('_', ' ')})"})
    return pd.DataFrame(rows)


@st.cache_data(ttl=EVERY_H * 3600)
def watch_now(t):
    return Watcher(tr, cfg).tick(t)


# ---- sidebar ------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="hs-brand"><span style="color:#d03b3b">&#9829;</span> HeartSafe</p>'
                '<p class="hs-brand-sub">Safety and follow-up layer for remote heart-failure monitoring</p>',
                unsafe_allow_html=True)
    role = st.selectbox("Signed in as", ["clinician", "administrator", "patient"], format_func=str.capitalize)
    if role == "patient":
        user = st.selectbox("Patient", ["DEMO-MARY"] + sorted(p for p in patients.patient_id if p != "DEMO-MARY"),
                            format_func=lambda p: "DEMO-MARY · demo patient" if p == "DEMO-MARY" else p)
    else:
        user = st.text_input("Name", "nurse_demo" if role == "clinician" else "admin_demo")
    st.markdown(f'<div class="hs-clock">Simulation clock: {day(now)}<br>Simulated data, modelled on Medly</div>',
                unsafe_allow_html=True)


# ---- one case -----------------------------------------------------------------------------------
def vitals_chart(pid, upto_day=None, window=21, measure="weight_kg"):
    titles = {"weight_kg": "Weight (kg)", "sbp": "Systolic BP (mmHg)", "hr": "Heart rate (bpm)",
              "sleep_hours": "Sleep (hours a night)"}
    f = feed[(feed.patient_id == pid) & feed.submitted]
    if upto_day is not None:
        f = f[(f.day <= upto_day) & (f.day > upto_day - window)]
    f = f[f[measure].notna()]
    d = decisions[decisions.patient_id == pid].copy()
    d["action"] = d.decision.map(lambda x: DECISION.get(x, (x, ""))[0])
    d["group"] = d.decision.map(lambda x: DECISION.get(x, ("", "Follow-up and patient loop"))[1])
    d = d[d.group.isin(GROUPS.domain)]  # routine ECG checks are in the table, not on the chart
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
            if describe_selfcare(review.get("selfcare")):
                label("Sleep and breathing (logged by the patient)")
                text(describe_selfcare(review["selfcare"]))
            label("Suggested next step")
            st.info(review["next_step"])
            st.caption("HeartSafe does not make the clinical decision; the clinical team does.")
        with right:
            first = "Sleep" if case["kind"] == "sleep_check" and "Sleep" in MEASURES else "Weight"
            m = st.segmented_control("Measure", list(MEASURES), default=first,
                                     key=f"m{case['case_id']}", label_visibility="collapsed")
            st.altair_chart(style(vitals_chart(case["patient_id"], review["day"], measure=MEASURES[m or first]),
                                  240), width="stretch")
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
        m = st.segmented_control("Measure", list(MEASURES), default="Weight", key="pj_measure")
        st.altair_chart(style(vitals_chart(pid, measure=MEASURES[m or "Weight"]), 280), width="stretch")
        st.caption("Gray line: daily readings. Coloured dots: days HeartSafe acted (hover for details).")

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

    hist = selfcare_history(pid)
    sc = describe_selfcare({"sleep": sleep_status(hist, None, cfg), "breathing": breathing_status(hist, None, cfg)})
    logged = logged_table(pid)
    if sc or len(logged):
        st.subheader("Sleep and breathing (logged by the patient)")
        text(sc or "No sleep or breathing entries from the platform.")
        if len(logged):
            st.dataframe(logged, hide_index=True, width="stretch", height=min(36 * len(logged) + 40, 220))
            st.caption("Entries from the patient's Sleep and breathing page. Worse sleep alone is a routine check-in "
                       "call, never a clinical alert.")

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

    t1, t2, t_rh, t_sc, t3, t4 = st.tabs(["Speed and follow-up", "Patient loop", "Heart rhythm", "Sleep and breathing",
                                          "Model evidence", "Assumptions and raw data"])
    with t_rh:
        rh = r.get("rhythm")
        if not rh:
            st.info("No rhythm results in this run. Re-run the cohort:  python -m heartsafe.run")
        else:
            c = st.columns(4)
            c[0].metric("New AF spells reported", f"{rh['spells_reported_to_care_team']:.0%}",
                        help=f"{rh['new_af_spells_in_patients_with_ecg_kit']} first spells of atrial fibrillation in "
                             "patients who have the ECG device")
            c[1].metric("Median days from onset to report", f"{rh['median_days_onset_to_report']:g}")
            c[2].metric("AF named on AF recordings", f"{rh['af_named_on_reliable_af_recordings']:.0%}",
                        help="Reliable recordings whose true rhythm was atrial fibrillation")
            c[3].metric("Heart-rate spells confirmed on the ECG", f"{rh['heart_rate_spells_confirmed_on_the_ecg']:.0%}",
                        help="Spells of a heart rate outside the patient's own limits, in patients with the ECG device")
            a, b2 = st.columns([3, 2], gap="large")
            sev, em = rh["new_af_report_severity"], rh["emergencies_where_the_ecg_finding_was_known_af"]
            with a:
                st.markdown("**What HeartSafe did with the named rhythm**")
                st.dataframe(pd.DataFrame([
                    ("Reliable ECG recordings read", rh["reliable_recordings"]),
                    ("Recordings named atrial fibrillation", rh["recordings_named_af"]),
                    ("Known AF: logged, not escalated as new", rh["known_af_recordings_not_escalated_as_new"]),
                    ("New AF reported to the care team", rh["new_af_reports"]),
                    ("↳ with symptoms (critical)", sev.get("critical", 0)),
                    ("↳ without symptoms (caution)", sev.get("caution", 0)),
                    ("↳ raised to critical when symptoms followed", rh["raised_to_critical_when_symptoms_followed"]),
                    ("Later AF recordings added to the same report", rh["later_af_recordings_already_reported"]),
                    ("Slow heart rate on the ECG (below the patient's limit)", rh["bradycardia_on_ecg"]),
                    ("Fast heart rate on the ECG", rh["tachycardia_on_ecg"]),
                    ("Platform heart-rate alerts confirmed on the ECG",
                     rh["alerts_with_the_rate_confirmed_on_the_ecg"]["alerts"]),
                ], columns=["Step", "Count"]), hide_index=True, width="stretch", height=425)
            with b2:
                st.warning(f"**{rh['new_af_reports_on_a_recording_without_af']} of {rh['new_af_reports']} new-AF "
                           "reports** named AF on a recording whose true rhythm was not AF. The rhythm model is a "
                           "single-lead screen: every report asks the clinician to confirm on a 12-lead ECG.")
                st.markdown("**Two questions for the cardiologist** (switches in config.yaml, `ecg_rhythm`)")
                text(f"• <b>known_af_triggers_emergency</b> (now on): {em['cases']} of {em['of_all_emergencies']} "
                     f"emergencies had known AF as the ECG finding; the clinician cancelled {em['clinician_cancelled']} "
                     "of them. Should known AF with chest pain or fainting still count as an abnormal ECG?")
                text(f"• <b>known_af_upgrades_with_symptoms</b> (now off): "
                     f"{rh['caution_alerts_with_known_af_on_ecg_not_raised']} caution alerts stayed caution because "
                     "the only ECG finding was AF the patient is already known to have.")
    with t_sc:
        sc = r.get("selfcare")
        if not sc:
            st.info("No sleep or breathing results in this run. Re-run the cohort:  python -m heartsafe.run")
        else:
            br = sc["breathing"]
            c = st.columns(4)
            c[0].metric("Sleep check-ins / 100 patient-months", f"{sc['checkins_per_100_patient_months']:g}",
                        help="Low-priority calls for the nurse; never a clinical alert")
            c[1].metric("Poor-sleep spells with a check-in", f"{sc['spells_with_checkin_or_note']:.0%}",
                        help=f"{sc['poor_sleep_spells']} simulated spells of poor sleep")
            c[2].metric("Median hours to the check-in call", f"{sc['median_hours_to_checkin_call']:g}")
            c[3].metric("Days meeting the breathing plan", f"{br['days_meeting_plan']:.0%}",
                        help=f"Plan: {br['plan_sessions_per_day']} sessions a day")
            a, b2 = st.columns(2, gap="large")
            with a:
                st.markdown("**What was behind the sleep check-ins**")
                why = pd.DataFrame([{"reason": k.capitalize(), "n": v} for k, v in sc["what_was_behind_them"].items()])
                st.altair_chart(style(alt.Chart(why).mark_bar(color=BLUE, cornerRadiusTopRight=4,
                                                              cornerRadiusBottomRight=4, size=22).encode(
                    y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                    x=alt.X("n:Q", title="Check-ins and notes"),
                    tooltip=[alt.Tooltip("reason:N", title="Behind it"), alt.Tooltip("n:Q", title="Count")]), 150),
                    width="stretch")
                st.caption(f"{sc['sleep_checkins']} check-in calls and {sc['notes_added_to_open_case']} notes added to "
                           "a case that was already open. Sleep is compared with the patient's own usual night.")
            with b2:
                st.markdown("**Sleep and breathing logged**")
                st.dataframe(pd.DataFrame([
                    ("Nights logged", f"{sc['nights_logged']:,}"),
                    ("Median sleep", f"{sc['median_sleep_h']:g} h"),
                    ("Breathing sessions a day (plan)", f"{br['mean_sessions_per_day']:g} ({br['plan_sessions_per_day']})"),
                    ("Patients doing under half the plan", f"{br['patients_below_half_plan']:.0%}"),
                ], columns=["Measure", "Value"]), hide_index=True, width="stretch")
                st.caption("Sleep, poor-sleep spells and breathing adherence are simulated (config.yaml). The breathing "
                           "plan is the care team's; HeartSafe only counts the sessions the patient logs.")
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
                 "HeartSafe": f"{q['heartsafe_clinical_cases'] + q['heartsafe_nurse_verifications'] + q['heartsafe_patient_followups'] + q.get('heartsafe_sleep_checkins', 0):.0f}",
                 "Platform alone": f"{q['platform_alone_alert_items']:.0f}"}]), hide_index=True, width="stretch")
            if q.get("heartsafe_sleep_checkins"):
                st.caption(f"The HeartSafe queue includes {q['heartsafe_sleep_checkins']:g} routine sleep check-in "
                           "calls per 100 patient-months, which the platform alone does not have.")
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
        rhy = jload("ecg/results/rhythm_evaluation.json")
        if rhy:
            st.markdown("**Naming the rhythm: normal, atrial fibrillation (AF) or another abnormal rhythm**")
            st.dataframe(pd.DataFrame([
                {"Test set (recordings the quality check accepted)": name, "Recordings": rhy[k]["n"],
                 "AF caught (sensitivity)": f"{rhy[k]['af']['sensitivity']:.0%}",
                 "Not AF, not called AF (specificity)": f"{rhy[k]['af']['specificity']:.1%}",
                 "Called AF and is AF (PPV)": f"{rhy[k]['af']['ppv']:.0%}", "AF AUROC": f"{rhy[k]['af']['auroc']:.3f}",
                 "3-class accuracy": f"{rhy[k]['accuracy']:.0%}"}
                for k, name in (("ptbxl_test_accepted", "PTB-XL test, lead I"),
                                ("cinc2017_test_accepted", "New device (AliveCor, PhysioNet 2017)")) if k in rhy]),
                hide_index=True, width="stretch")
            hr_ = rhy.get("heart_rate_from_r_peaks") or {}
            rows = [{"Rhythm in the cardiologist's report": name, "Recordings": hr_[k]["n"],
                     "Caught by the heart rate": f"{hr_[k]['sensitivity']:.0%}",
                     "Specificity": f"{hr_[k]['specificity']:.0%}", "Median heart rate (bpm)": f"{hr_[k]['median_hr_bpm']:g}"}
                    for k, name in (("SBRAD", "Sinus bradycardia (slow)"), ("STACH", "Sinus tachycardia (fast)"))
                    if k in hr_]
            if rows:
                st.markdown("**Slow and fast heart rate, from the R-peaks the quality check already finds**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("In the agent the heart rate is compared with the patient's own limits from enrollment, "
                           "not with a fixed 60 / 100 bpm.")
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
         "Is the ECG good enough to trust? Are the MRI measurements reliable? Then: accept, repeat, or escalate. "
         "A reliable ECG is named: new atrial fibrillation is reported, known AF is only logged, and the heart "
         "rate is checked against the patient's own limits."),
        ("4 · Tracked cases", "Each escalation becomes a case with an explainable review. Deadlines trigger reminders "
         "and backup routing; a follow-up watch re-checks every case every few hours."),
        ("5 · Patient loop", "After the clinician decides, the instruction goes to the patient; a caregiver or "
         "community health worker is contacted if they do not confirm. Readings are checked after the action."),
        ("6 · Equity monitor", "Compares groups (remoteness, sex, age, device) and proposes safeguards, each with its "
         "cost, that an administrator approves."),
        ("7 · Sleep and breathing", "The patient logs last night's sleep and the breathing exercises from the care "
         "team's plan. Sleep worse than their own usual opens a routine check-in call, never a clinical alert; "
         "breathlessness at night is passed on at caution level. Both show in every review."),
    ]
    for row in (steps[:4], steps[4:]):
        cols = st.columns(4)
        for col, (t, body) in zip(cols, row):
            with col.container(border=True, height=330):
                st.markdown(f"**{t}**")
                text(body)
    st.info("Only a clinician acknowledges, reviews and resolves a case, and only the patient side confirms an "
            "instruction. Every action is in the audit log.")


# ---- the patient's page: sleep tracker and breathing exercises ------------------------------------
# A guided breathing timer. The circle grows on the breath in and shrinks on the breath out; with
# "reduce motion" set on the device it stays still and only the words and the countdown change.
TIMER_HTML = """
<style>
  html, body {margin: 0; background: transparent; color: #0b0b0b;
    font-family: "Source Sans 3", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif;}
  .wrap {display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 6px 0 0;}
  .stage {width: 230px; height: 230px; position: relative; display: flex; align-items: center; justify-content: center;}
  .ring {position: absolute; inset: 0; border-radius: 50%; border: 2px dashed #c3c2b7;}
  .ball {position: absolute; inset: 4px; border-radius: 50%; background: #cde2fb; border: 2px solid #2a78d6;
    transform: scale(0.55); transition: transform 1s ease-in-out;}
  .words {position: relative; text-align: center;}
  .phase {font-size: 1.2rem; font-weight: 700;}
  .sec {font-size: 2.1rem; font-weight: 700; color: #184f95; font-variant-numeric: tabular-nums; min-height: 2.5rem;}
  .meta {color: #52514e; font-size: 0.95rem; min-height: 1.3rem; text-align: center;}
  button {font: inherit; font-size: 0.95rem; padding: 7px 18px; border-radius: 8px; cursor: pointer;
    border: 1px solid #c3c2b7; background: #fcfcfb; color: #0b0b0b; margin: 0 3px;}
  button.primary {background: #2a78d6; border-color: #2a78d6; color: #fff; font-weight: 600;}
  button:focus-visible {outline: 3px solid #184f95; outline-offset: 2px;}
  @media (prefers-reduced-motion: reduce) {.ball {transition: none !important; transform: scale(0.8) !important;}}
</style>
<div class="wrap">
  <div class="stage">
    <div class="ring"></div><div class="ball" id="ball"></div>
    <div class="words"><div class="phase" id="phase" aria-live="polite">Ready</div><div class="sec" id="sec"></div></div>
  </div>
  <div class="meta" id="meta"></div>
  <div><button class="primary" id="go">Start</button><button id="again">Restart</button></div>
</div>
<script>
  const PHASES = __PHASES__, TOTAL = __TOTAL__;            // [words, seconds, circle size], session seconds
  const CYCLE = PHASES.reduce((a, p) => a + p[1], 0), BREATHS = Math.floor(TOTAL / CYCLE), END = BREATHS * CYCLE;
  const ball = document.getElementById("ball"), phase = document.getElementById("phase"),
        sec = document.getElementById("sec"), meta = document.getElementById("meta"),
        go = document.getElementById("go"), again = document.getElementById("again");
  let t = 0, timer = null, shown = -1;
  const clock = s => Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  function where() {                                         // the phase at second t, and seconds left in it
    let c = t % CYCLE, i = 0;
    while (c >= PHASES[i][1]) { c -= PHASES[i][1]; i++; }
    return [i, PHASES[i][1] - c];
  }
  function move(i, seconds) {                                // send the circle to the size of phase i
    ball.style.transition = "transform " + seconds + "s ease-in-out";
    ball.style.transform = "scale(" + PHASES[i][2] + ")";
  }
  function draw() {
    if (t >= END) {
      stop(); phase.textContent = "Well done"; sec.textContent = ""; go.disabled = true; go.textContent = "Start";
      move(PHASES.length - 1, 1);
      meta.textContent = "Session finished. Press \\u201cI finished this session\\u201d below to log it.";
      return;
    }
    const [i, left] = where();
    if (timer && i !== shown) { move(i, left); shown = i; }
    phase.textContent = t === 0 && !timer ? "Ready" : PHASES[i][0];
    sec.textContent = t === 0 && !timer ? "" : left;
    meta.textContent = clock(END - t) + " left \\u00b7 breath " + (Math.floor(t / CYCLE) + 1) + " of " + BREATHS;
  }
  function stop() { clearInterval(timer); timer = null; }
  go.onclick = () => {
    if (timer) {                                             // pause: freeze the circle where it is
      stop(); ball.style.transform = getComputedStyle(ball).transform; ball.style.transition = "none";
      shown = -1; go.textContent = "Continue"; return;
    }
    go.textContent = "Pause";
    timer = setInterval(() => { t += 1; draw(); }, 1000);
    draw();
  };
  again.onclick = () => {
    stop(); t = 0; shown = -1; go.disabled = false; go.textContent = "Start";
    ball.style.transition = "transform 0.6s ease-in-out"; ball.style.transform = "scale(0.55)"; draw();
  };
  draw();
</script>
"""


def breathing_timer(ex, minutes):
    phases = [p for p in (["Breathe in", ex["in_s"], 1.0], ["Hold", ex["hold_s"], 1.0],
                          ["Breathe out", ex["out_s"], 0.55]) if p[1] > 0]
    # TIMER_HTML is a constant of this file; only numbers from config.yaml are put into it
    st.iframe(TIMER_HTML.replace("__PHASES__", json.dumps(phases)).replace("__TOTAL__", str(int(minutes * 60))),
              height=350)


def sleep_chart(hist, usual):
    d = pd.DataFrame([{"day": h["day"], "hours": h["sleep_hours"], "woke": h.get("sleep_awakenings")}
                      for h in hist if pd.notna(h.get("sleep_hours"))])
    chart = alt.Chart(d).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=20).encode(
        x=alt.X("day:O", title="Morning of day", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("hours:Q", title="Hours slept", scale=alt.Scale(domain=[0, max(10, d.hours.max())])),
        tooltip=[alt.Tooltip("day:O", title="Morning of day"), alt.Tooltip("hours:Q", title="Hours slept", format=".1f"),
                 alt.Tooltip("woke:Q", title="Times woken")])
    if usual is not None:
        chart = chart + alt.Chart(pd.DataFrame({"usual": [usual]})).mark_rule(
            strokeDash=[5, 4], strokeWidth=2, color=INK2).encode(
            y="usual:Q", tooltip=[alt.Tooltip("usual:Q", title="Your usual night (h)", format=".1f")])
    return chart


def page_selfcare():
    pid = user
    header("Sleep and breathing", "Log last night's sleep and do the breathing exercises from your care team's plan. "
                                  "Your care team sees these entries next to your other readings.")
    flash = st.session_state.pop("sc_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    hist = selfcare_history(pid)
    before = [h for h in hist if h["day"] < TODAY]
    today = next((h for h in hist if h["day"] == TODAY), None) or {"day": TODAY}
    slept = today.get("sleep_hours")
    sl = sleep_status(before, today, cfg) or {}
    usual = sl.get("usual_h")
    if usual is None:  # fewer than the nights the comparison needs: show the median of what there is
        earlier = [h["sleep_hours"] for h in before if pd.notna(h.get("sleep_hours"))]
        usual_shown = round(float(pd.Series(earlier).median()), 1) if earlier else None
    else:
        usual_shown = usual
    br = cfg["selfcare"]["breathing"]
    plan, done = br["sessions_per_day"], int(today.get("breathing_sessions") or 0)
    c = st.columns(4)
    c[0].metric("Last night", f"{slept:g} h" if slept is not None else "Not logged yet")
    c[1].metric("Your usual night", f"{usual_shown:g} h" if usual_shown is not None else "—",
                help="The median of your earlier nights. Your sleep is compared with your own usual, not with a "
                     "fixed number of hours.")
    c[2].metric(f"Average, last {sl.get('nights', 0)} night(s)", f"{sl['recent_h']:g} h" if sl else "—")
    c[3].metric("Breathing sessions today", f"{done} of {plan}", help="The number of sessions is your care team's plan")

    t_sleep, t_breath, t_mine = st.tabs(["Sleep", "Breathing exercises", "My entries"])
    with t_sleep:
        left, right = st.columns([2, 3], gap="large")
        with left, st.form("sleep_form"):
            st.markdown("**Last night**")
            hours_ = st.number_input("Hours slept", 0.0, 24.0, float(slept if slept is not None else usual_shown or 7.0),
                                     step=0.5)
            woke = st.number_input("Times you woke up", 0, 30, int(today.get("sleep_awakenings") or 0), step=1)
            propped = st.checkbox("I slept propped up or sitting, because lying flat made me short of breath",
                                  value=bool(today.get("propped_up")))
            breathless = st.checkbox("I woke up short of breath", value=bool(today.get("woke_breathless")))
            kept = st.pills("What kept you awake? (optional)", KEPT_AWAKE, selection_mode="multi",
                            default=today.get("kept_awake") or None, format_func=str.capitalize)
            if st.form_submit_button("Save last night's sleep", type="primary"):
                sc_log.log_sleep(pid, TODAY, now, pid, "patient", hours_, woke, propped, breathless, kept or [])
                entry = {**today, "sleep_hours": float(hours_), "sleep_awakenings": int(woke),
                         "propped_up": propped, "woke_breathless": breathless}
                patient = patients.set_index("patient_id", drop=False).to_dict("index")[pid]
                what, _, _ = checkin_from_sleep_log(tr, cfg, patient, before, entry, now)
                ack = cfg["followup"]["ack_deadline_h"]
                if what == "checkin" and (propped or breathless):
                    flash = ("warning", "Saved. You reported being short of breath at night, so your care team has "
                             f"been told and a nurse is asked to call you within {ack['caution']:g} hours. "
                             + cfg["emergency"]["patient_message"])
                elif what == "checkin":
                    flash = ("info", "Saved. Your sleep has been worse than your usual for a few nights, so a nurse "
                             f"is asked to call you for a routine check-in within {ack['low'] / 24:g} days.")
                elif what == "noted":
                    flash = ("info", "Saved, and added to the case your care team already has open for you."
                             + (" " + cfg["emergency"]["patient_message"] if propped or breathless else ""))
                else:
                    flash = ("success", "Saved. Thank you.")
                st.session_state["sc_flash"] = flash
                st.rerun()
        with right:
            st.markdown("**Your last nights**")
            if any(pd.notna(h.get("sleep_hours")) for h in hist):
                st.altair_chart(style(sleep_chart(hist, usual_shown), 270), width="stretch")
                st.caption("Bars: hours slept each night"
                           + (f". Dashed line: your usual night ({usual_shown:g} h)." if usual_shown is not None else "."))
            else:
                st.info("No nights logged yet. Your first entry starts the chart.")
        st.caption(f"Demo: the simulation clock is fixed at day {TODAY}, so an entry is saved for this morning; saving "
                   "again replaces it. Sleep is not monitored in real time.")
    with t_breath:
        exs = br["exercises"]
        left, right = st.columns([3, 2], gap="large")
        with left:
            a, b = st.columns([3, 2])
            key = a.radio("Exercise", list(exs), format_func=lambda k: (
                f"{exs[k]['name']}: in {exs[k]['in_s']} s, "
                + (f"hold {exs[k]['hold_s']} s, " if exs[k]["hold_s"] else "") + f"out {exs[k]['out_s']} s"))
            minutes = b.segmented_control("Length", [2, 5], default=2, format_func=lambda m: f"{m} minutes") or 2
            breathing_timer(exs[key], minutes)
            if st.button("I finished this session", type="primary"):
                sc_log.log_breathing(pid, TODAY, now, pid, "patient", exs[key]["name"], minutes)
                left_ = plan - done - 1
                st.session_state["sc_flash"] = ("success", "Session logged. " + (
                    f"{left_} more to go today." if left_ > 0 else "You have done today's plan."))
                st.rerun()
        with right:
            label("Today")
            st.progress(min(done / plan, 1.0), text=f"{done} of {plan} sessions in your care team's plan")
            week = [h for h in hist if pd.notna(h.get("breathing_sessions"))][-7:]
            if week:
                label("Last 7 days")
                w = pd.DataFrame([{"day": h["day"], "sessions": h["breathing_sessions"]} for h in week])
                st.altair_chart(style(alt.Chart(w).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                                            size=20).encode(
                    x=alt.X("day:O", title="Day", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("sessions:Q", title="Sessions", axis=alt.Axis(tickMinStep=1),
                            scale=alt.Scale(domain=[0, max(plan, w.sessions.max())])),
                    tooltip=[alt.Tooltip("day:O", title="Day"), alt.Tooltip("sessions:Q", title="Sessions")])
                    + alt.Chart(pd.DataFrame({"plan": [plan]})).mark_rule(strokeDash=[5, 4], strokeWidth=2, color=INK2)
                    .encode(y="plan:Q", tooltip=[alt.Tooltip("plan:Q", title="Plan (sessions a day)")]), 150),
                    width="stretch")
                st.caption(f"Dashed line: the plan ({plan} sessions a day).")
            with st.container(border=True):
                st.markdown("**Before you start**")
                text("• Sit upright with your back supported and your shoulders relaxed.<br>"
                     "• Breathe gently. Never force a breath or hold it until it is uncomfortable.<br>"
                     "• After chest surgery, hold a pillow firmly against your chest if you need to cough.<br>"
                     "• <b>Stop and rest</b> if you feel dizzy, get chest pain, or become more short of breath.")
            st.caption("Which exercises to do and how often is your care team's plan; follow their instructions if "
                       "they differ from this page. Nobody watches these sessions in real time. "
                       + cfg["emergency"]["patient_message"])
    with t_mine:
        logged = logged_table(pid)
        if len(logged):
            st.dataframe(logged, hide_index=True, width="stretch")
        else:
            st.info("Nothing logged in the app yet.")
        st.caption("Entries are kept as they were logged; a new sleep entry for the same night replaces the earlier "
                   "one in the chart. Your care team can read them, and only you, a caregiver or a community health "
                   "worker can add them.")


if role == "patient":
    pages = [st.Page(page_selfcare, title="Sleep and breathing", icon=":material/bedtime:", default=True)]
else:
    pages = [
        st.Page(page_queue, title="Clinician queue", icon=":material/inbox:", default=True),
        st.Page(page_patient, title="Patient journey", icon=":material/person:"),
        st.Page(page_equity, title="Equity audit", icon=":material/balance:"),
        st.Page(page_results, title="Results", icon=":material/monitoring:"),
        st.Page(page_models, title="Model reliability", icon=":material/verified:"),
        st.Page(page_how, title="How it works", icon=":material/help:"),
    ]
st.navigation(pages).run()
