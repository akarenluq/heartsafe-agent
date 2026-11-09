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
(columns: patient_id, day, submitted, weight_kg, sbp, hr, sob, swelling; patients:
patient_id, remoteness, sex, age, age_band, device, ecg_kit).

## What the agent does with each platform alert

| Check | Example | Action |
|---|---|---|
| Implausible value, weight in lb, BP cuff artifact | 154 kg after 70 kg | ask for a repeat, no case |
| Overnight weight jump with no symptoms | +2.6 kg, stable HR | ask for a repeat; if it persists, re-baseline (new scale) |
| Alert no longer stands on corrected data | 7-day gain explained by scale change | filtered, logged |
| ECG quality or model uncertainty too poor | quality 0.4 | ECG not used as evidence, repeat ECG requested |
| Reliable abnormal ECG with symptoms | caution alert | raised to critical |
| Otherwise | | tracked case with an explainable review |
| 3 days with no readings | | outreach task |
| MRI result flagged by the imaging pipeline | high uncertainty | imaging review case |

Cases go created -> acknowledged -> reviewed -> resolved. Missed deadlines send
reminders, a second miss goes to a backup clinician, and only a clinician can move
or close a case (`tracker.py`). Every action is in the audit log.

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
