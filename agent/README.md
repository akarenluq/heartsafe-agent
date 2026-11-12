# HeartSafe coordination layer

HeartSafe is an equity-aware trust and coordination layer on top of an
existing remote-monitoring platform. Our main reference is Medly (UHN), which is
used in remote and Indigenous communities; TELUS Home Health Monitoring and Cloud DX
work the same way. The platform collects daily weight, blood pressure, heart rate
and symptoms and raises alerts; HeartSafe checks whether each
alert rests on reliable data, prepares an explainable review, tracks every
escalation until a clinician resolves it, and monitors equity across communities.
It complements the platform and scheduled follow-up visits; it does not replace them.

```
Platform feed     -> HeartSafe agent -> clinical team (acknowledge, review, decide) -> tracked follow-up
                    |  reliability checks (readings, ECG, MRI)
                    |  explainable review
                    |  equity monitor -> safeguards approved by an administrator
```

**These platforms are not open source.** No public code or API was found for Medly,
so `PlatformAdapter` reads a generic daily CSV and the rules in `monitoring.py`
approximate a normal / caution / critical alert modelled on Medly's messages. Any
platform that can export the columns below can be plugged in. The cohort is simulated
(2,000 patients x 90 days by default); every simulation number is an assumption in
`config.yaml` and is the same for HeartSafe and the platform-alone baseline.

## Run it

```bash
pip install -r requirements.txt
python -m heartsafe.run            # simulate, run agent + baseline, evaluate, equity audit -> data/
streamlit run app.py               # dashboard
pytest                             # tests
```

Real model outputs are used automatically when `../ecg/results` and `../imaging/results`
exist (clean + fault-injected sets; copy them from SDRE with `bash ../fetch_results.sh`).
The agent then uses each pipeline's own calibrated decision (`suggested_action`, `flags`);
`--simulated-models` ignores them, and `--ecg` / `--imaging` take other folders (comma-separated).
Which patients get a faulty ECG or a scan on an unfamiliar scanner is a simulation
assumption (`ecg_fault_share`, `regional_scanner_share`, `mri_fault_share` in `config.yaml`).

Safeguard loop: an administrator approves a proposal on the Equity audit page and presses
"Apply approved safeguards now"; the cohort is re-run and the page shows the extra cases,
false alarms and reminders against the last run without safeguards.

Use a real platform export (e.g. from Medly) instead of the simulation:
`python -m heartsafe.run --feed export.csv --patients-csv patients.csv`
(columns: patient_id, day, submitted, weight_kg, sbp, hr, sob, swelling; optional: dbp, spo2,
orthopnea, pnd, chest_pain, syncope, diuretic_taken, potassium, creatinine_umol, sleep_hours,
sleep_awakenings, breathing_sessions; patients: patient_id, remoteness, sex, age, age_band, device,
ecg_kit; optional: nyha, caregiver, creatinine_baseline_umol and the patient's own limits sbp_low,
sbp_high, dbp_high, hr_low, hr_high).

The run builds the case database in memory and saves it at the end, so a re-run works while the
dashboard is open. A full run (2,000 patients, real model outputs) takes about 6 minutes on a laptop.

## Who is enrolled

Reviewed with cardiologists (Nov 2026), adapted from the Medly post-discharge trial
(NCT03358303). Settings are under `inclusion` in `config.yaml`; the simulation screens
candidates and writes the counts to `data/screening.json`.

- Adults hospitalised more than 48 h for decompensated heart failure, discharged in the last 14 days
- NYHA class II-IV (assessed at the bedside; ejection fraction is not required because it needs an echocardiogram)
- Able to take readings and answer the symptom questions, alone or with a caregiver
- Excluded: dementia or uncontrolled psychiatric illness, long-term care, life expectancy under 1 year,
  transplant list or ventricular assist device, dialysis
- Target communities are rural and remote; urban patients stay in the simulation as the equity comparison group

## Platform alert rules

Caution: weight +1.5 kg in 2 days or +2.5 kg in 7 days, shortness of breath, swelling, breathless lying
flat, waking up breathless, diuretic not taken 2 days in a row, oxygen saturation below 93%, potassium
outside 3.5-5.5 mmol/L, creatinine 26.5 umol/L (0.3 mg/dL) above the enrollment value.
Critical: weight +3.5 kg in 7 days, systolic BP below 90 or above 180, diastolic BP above 110, heart
rate below 45 or above 120, oxygen saturation below 90%, chest pain, fainting, potassium outside
3.0-6.0 mmol/L, or breathlessness together with a weight-gain rule. BP and heart-rate limits are
defaults; the clinician can set a patient's own limits at enrollment.

**Emergency** (HeartSafe level, above critical; cardiologist review, Nov 2026): chest pain or fainting
**and** a reliable abnormal ECG at the same time. The patient is told at once: "If you feel worse, call
911." A clinician must start a video call within 15 minutes (not 4 hours) and records "call 911" or
"cancel the alarm" with a reason (`tracker.emergency_decision`). HeartSafe never calls 911 itself.
Chest pain or fainting without an abnormal ECG stays critical. If the ECG is not usable, a repeat
recording is asked for straight away.

## ECG recordings

Every ECG the patient records is checked, not only those on alert days (`ecg_protocol` in
`config.yaml`: weekly, on symptom days, and when the agent asks on alert days). A recording that is
not usable (disconnected, noisy, low quality) gets "please record it again"; a reliable abnormal
recording with no alert is added to the chart for the next review; a normal one is logged.

A reliable recording is also named, not only called normal or abnormal (`ecg_rhythm` in `config.yaml`):

- **Atrial fibrillation (AF).** A second ECG model gives normal / AF / other abnormal
  (`../ecg/scripts/05_rhythm.py`). AF is named when its AF probability is at least 0.5; below that the
  rhythm is "unclear" and the recording is handled as before.
  - **New AF** (no AF in the enrollment record): reported to the care team once. With symptoms it is
    **critical**; without symptoms it is caution, and it is raised to critical if symptoms follow.
    Later AF recordings are logged as "already reported".
  - **Known AF** (AF in the enrollment record): logged, never reported as new.
- **Heart rate from the R-peaks** of the recording, against the patient's own limits (`hr_low`,
  `hr_high`; defaults 45 and 120): bradycardia or tachycardia. With no alert it is added to the chart;
  with a caution alert and symptoms the case is raised to critical.

Two choices about known AF are switches for the cardiologists to set, with what each cost in the last
simulated run:

| Switch (`ecg_rhythm`) | Default | In the simulated cohort |
|---|---|---|
| `known_af_triggers_emergency` | true: chest pain or fainting with known AF on the ECG is still an emergency | 31 of 74 emergencies had known AF as the ECG finding; clinicians cancelled 10 of them |
| `known_af_upgrades_with_symptoms` | false: a caution alert with symptoms is not raised to critical when the only ECG finding is known AF | 425 caution alerts stayed caution |

Rhythm model on held-out recordings it accepted: PTB-XL (n = 1,827) AF sensitivity 0.86, specificity
0.99; a second device, PhysioNet/CinC 2017 (n = 1,220), sensitivity 0.76, specificity 0.96. Heart rate
from the R-peaks against the PTB-XL statements: sinus tachycardia sensitivity 0.89, sinus bradycardia
0.63. In the simulated cohort, 29 of 31 new-AF spells in patients with an ECG kit reached the care
team (median 2 days from onset), and no recording from a patient with known AF was reported as new.
**Limit:** 20 of the 49 new-AF reports named AF on a recording whose simulated rhythm was not AF. The
report asks the clinician to confirm on a 12-lead ECG; waiting for a second recording before
reporting is possible future work. The platform's own heart-rate rule is already critical, so the
ECG heart rate mostly confirms an alert (31 of 45 simulated heart-rate spells) and does not add new ones.

## What the agent does with each platform alert

| Check | Example | Action |
|---|---|---|
| Chest pain or fainting with a reliable abnormal ECG | | **emergency**: message to the patient now, video call within 15 min, clinician decides 911 or cancel |
| Implausible value, weight in lb, BP cuff artifact | 154 kg after 70 kg | ask for a repeat, no case |
| Overnight weight jump with no symptoms | +2.6 kg, stable HR | ask for a repeat; if it persists, a nurse calls to verify a new scale. Weight-only alerts wait in that case; symptoms or any other rule escalate at once |
| Nurse confirms a new scale | | readings re-baselined |
| Nurse finds the same scale | | the gain is real: tracked clinical case |
| Alert no longer stands on corrected data | 7-day gain explained by a verified new scale | filtered, logged |
| ECG quality or model uncertainty too poor | quality 0.4 | ECG not used as evidence, repeat ECG requested |
| Reliable abnormal ECG with symptoms | caution alert | raised to critical (not when the only ECG finding is known AF, see above) |
| New AF on a reliable ECG | no AF at enrollment | reported once: critical with symptoms, caution without |
| Heart rate on the ECG outside the patient's own limits, with symptoms | caution alert | raised to critical |
| Sleep worse than the patient's own usual, no alert | 3 short nights | low-priority check-in call for the nurse, never a clinical alert |
| Otherwise | | tracked case with an explainable review |
| 3 days with no readings | | outreach task |
| MRI result flagged by the imaging pipeline | high uncertainty | imaging review case |

Cases go created -> acknowledged -> reviewed -> resolved. Missed deadlines send
reminders, a second miss goes to a backup clinician, and only a clinician can move
or close a case (`tracker.py`). Every action is in the audit log.

## Patient loop

After a clinician resolves a clinical case, HeartSafe follows the instruction to the patient
(`loops` table in `tracker.py`, settings under `patient_loop` in `config.yaml`):

1. **Notified.** The instruction is sent (app, text or call).
2. **Confirmed.** If the patient does not confirm in time (2 h critical, 12 h caution), HeartSafe
   contacts their caregiver, or the community health worker if there is none, then the community
   health worker.
3. **Action done.** If the action is not recorded in time, HeartSafe reminds whoever confirmed.
4. **Readings checked** 24 h after the action: improved -> loop closed; not improved -> a clinical
   case is reopened.

Only the patient, a caregiver or a community health worker can record a confirmation or an action;
the agent cannot. A patient who cannot be reached, or an instruction not completed, becomes a
`patient_followup` task for the nurse. The comparison arm ("platform alone") sends one message with
no follow-up. The equity audit compares the share of instructions completed across groups and can
propose contacting helpers earlier for a group (`earlier_patient_followup`, administrator-approved).
Patient response times and no-response rates are simulation assumptions, and the simulation does not
model the effect of treatment, so "readings did not improve" reopens are not a measure of care quality.

## Sleep and breathing exercises

Poor sleep and shallow breathing are common in the weeks after a heart-failure admission or heart
surgery. Patients log last night's sleep and their breathing-exercise sessions; HeartSafe shows both to
the care team and asks a nurse to call when sleep gets worse. It gives no medical advice: the breathing
plan comes from the care team, and the patient page says to stop and rest on dizziness, chest pain or
more shortness of breath. Settings are under `selfcare` in `config.yaml`.

- **Patient page ("Sleep and breathing", the patient role in the dashboard).** Sleep: hours slept,
  times woken, "slept propped up", "woke up short of breath", what kept them awake, and a chart
  against their own usual. Breathing exercises: a paced timer (slow breathing 4 s in / 6 s out, or
  with a 2 s pause), 2 or 5 minutes, counted against the plan (3 sessions a day by default).
- **Sleep rule.** Sleep is compared with the patient's own usual (the median of their earlier nights,
  at least 4), not with a fixed number of hours. "Worse" means each of the last 3 nights at least 1 h
  below usual, and on average at least 1.5 h below usual or under 4.5 h.
- **What the agent does.** Worse sleep on a day with no alert opens a low-priority **check-in call**
  for the nurse (`sleep_check`), at most one per patient in 21 days; with a case already open it is
  added to that case. It is never a clinical alert on its own. "Slept propped up" and "woke up short
  of breath" are symptoms the platform already alerts on, so those answers make the check-in caution.
- **In every review.** Sleep against usual and breathing sessions against the plan are shown on the
  case and on the patient journey.
- **Who can log.** Only the patient, a caregiver or a community health worker (`SelfCareLog`); entries
  made in the app are kept in `data/selfcare.db`, which a cohort re-run does not erase.

In the simulated cohort: 318 sleep check-ins and 50 notes on open cases (5.3 check-ins per 100
patient-months, against 89.5 clinical cases); 92% of simulated poor-sleep spells led to a check-in
or a note; breathing sessions averaged 2.0 a day against a plan of 3. How patients sleep and how often
they do the exercises are simulation assumptions (`simulation.selfcare` in `config.yaml`), so these
numbers show the workload of the rule, not a clinical benefit. In the demo the clock stands at day 90,
so a patient can log one night; the earlier nights come from the platform feed.

## Files

```
config.yaml            thresholds, deadlines, simulation assumptions
heartsafe/simulate.py  simulated cohort modelled on Medly (+ scripted demo patient DEMO-MARY)
heartsafe/monitoring.py  platform adapter and alert rules (modelled on Medly)
heartsafe/reliability.py  reading, ECG and MRI reliability checks
heartsafe/signals.py   ECG quality index; loaders for ECG / imaging pipeline outputs
heartsafe/selfcare.py  sleep rule, breathing-session count, the patient-side log
heartsafe/review.py    explainable review (template), ECG and MRI report summaries
heartsafe/watch.py     follow-up watch: checks every case and patient loop every few hours
heartsafe/orchestrator.py  optional Claude review writer (future work, off by default)
heartsafe/tracker.py   SQLite case tracker with audit log and permissions
heartsafe/clinicians.py  assumed clinician response model
heartsafe/agent.py     the agent and the platform-alone baseline
heartsafe/equity.py    subgroup audit with bootstrap CIs and safeguard proposals
heartsafe/evaluate.py  HeartSafe vs the platform alone on ground truth
heartsafe/run.py       command line entry point
app.py                 Streamlit dashboard
```

## Reviews and follow-up watch (no LLM needed)

Every case review is written from templates: why it was flagged, symptoms, the trend, a
plain-language summary of the ECG and MRI reports (what the model read, signal or image
quality, uncertainty, whether it was used as evidence, the ejection-fraction band), the
reliability caveats, the follow-up status and the suggested next step.

The follow-up watch (`heartsafe/watch.py`) checks every case and patient loop every
`followup.watch_every_h` hours (2 by default): missed acknowledgement or resolution
deadlines, reminders sent, and whether the patient side confirmed and acted. The dashboard's
clinician queue shows it and re-runs it on that interval. From the command line:

```
python -m heartsafe.watch                 # one check, report only
python -m heartsafe.watch --act --loop    # live: check every 2 h, remind the nurse or contact the next helper
```

It only reminds and contacts; acknowledging, resolving and confirming stay with people.

## LLM agent (optional, future work)

Off by default: the template review covers the same facts. To try it, set
`review.llm_enabled: true` in `config.yaml` and `ANTHROPIC_API_KEY`, then press "Explain with
AI agent" on a case (`python llm_check.py` tests 5 cases). Claude
(`review.llm_model` in `config.yaml`) calls read-only tools over the case (facts, data
quality, ECG, MRI, follow-up status) and writes the review through `submit_review`
(`heartsafe/orchestrator.py`). It explains; severity, actions and deadlines stay with the
deterministic policy and the clinical team. Every run goes to the case's audit log, and the
template review is shown if there is no key or the call fails.
