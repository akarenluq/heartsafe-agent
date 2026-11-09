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
orthopnea, pnd, chest_pain, syncope, diuretic_taken, potassium, creatinine_umol; patients:
patient_id, remoteness, sex, age, age_band, device, ecg_kit; optional: nyha, caregiver,
creatinine_baseline_umol and the patient's own limits sbp_low, sbp_high, dbp_high, hr_low, hr_high).

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

## What the agent does with each platform alert

| Check | Example | Action |
|---|---|---|
| Implausible value, weight in lb, BP cuff artifact | 154 kg after 70 kg | ask for a repeat, no case |
| Overnight weight jump with no symptoms | +2.6 kg, stable HR | ask for a repeat; if it persists, a nurse calls to verify a new scale. Weight-only alerts wait in that case; symptoms or any other rule escalate at once |
| Nurse confirms a new scale | | readings re-baselined |
| Nurse finds the same scale | | the gain is real: tracked clinical case |
| Alert no longer stands on corrected data | 7-day gain explained by a verified new scale | filtered, logged |
| ECG quality or model uncertainty too poor | quality 0.4 | ECG not used as evidence, repeat ECG requested |
| Reliable abnormal ECG with symptoms | caution alert | raised to critical |
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
4. **Readings checked** 72 h after the action: improved -> loop closed; not improved -> a clinical
   case is reopened.

Only the patient, a caregiver or a community health worker can record a confirmation or an action;
the agent cannot. A patient who cannot be reached, or an instruction not completed, becomes a
`patient_followup` task for the nurse. The comparison arm ("platform alone") sends one message with
no follow-up. The equity audit compares the share of instructions completed across groups and can
propose contacting helpers earlier for a group (`earlier_patient_followup`, administrator-approved).
Patient response times and no-response rates are simulation assumptions, and the simulation does not
model the effect of treatment, so "readings did not improve" reopens are not a measure of care quality.

## Files

```
config.yaml            thresholds, deadlines, simulation assumptions
heartsafe/simulate.py  simulated cohort modelled on Medly (+ scripted demo patient DEMO-MARY)
heartsafe/monitoring.py  platform adapter and alert rules (modelled on Medly)
heartsafe/reliability.py  reading, ECG and MRI reliability checks
heartsafe/signals.py   ECG quality index; loaders for ECG / imaging pipeline outputs
heartsafe/review.py    explainable review (template)
heartsafe/orchestrator.py  Claude tool-calling review writer with template fallback
heartsafe/tracker.py   SQLite case tracker with audit log and permissions
heartsafe/clinicians.py  assumed clinician response model
heartsafe/agent.py     the agent and the platform-alone baseline
heartsafe/equity.py    subgroup audit with bootstrap CIs and safeguard proposals
heartsafe/evaluate.py  HeartSafe vs the platform alone on ground truth
heartsafe/run.py       command line entry point
app.py                 Streamlit dashboard
```

## LLM agent (optional)

Set `ANTHROPIC_API_KEY` and press "Explain with AI agent" on a case. Claude
(`review.llm_model` in `config.yaml`) calls read-only tools over the case (facts, data
quality, ECG, MRI, follow-up status) and writes the review through `submit_review`
(`heartsafe/orchestrator.py`). It explains; severity, actions and deadlines stay with the
deterministic policy and the clinical team. Every run goes to the case's audit log, and the
template review is shown if there is no key or the call fails.
