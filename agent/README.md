# HeartSafe coordination layer

HeartSafe is an equity-aware trust and coordination layer on top of a
remote-monitoring platform such as Medly (UHN). Medly collects daily weight, blood
pressure, heart rate and symptoms and raises alerts; HeartSafe checks whether each
alert rests on reliable data, prepares an explainable review, tracks every
escalation until a clinician resolves it, and monitors equity across communities.
It complements Medly and scheduled follow-up visits; it does not replace them.

```
Medly-style feed -> HeartSafe agent -> clinical team (acknowledge, review, decide) -> tracked follow-up
                    |  reliability checks (readings, ECG, MRI)
                    |  explainable review
                    |  equity monitor -> safeguards approved by an administrator
```

**Medly is not open source.** No public code or API was found, so `MedlyAdapter`
reads a Medly-style CSV and the rules in `medly.py` approximate Medly's
normal / caution / critical messages. The cohort is simulated
(2,000 patients x 90 days by default); every simulation number is an assumption in
`config.yaml` and is the same for HeartSafe and the Medly-alone baseline.

## Run it

```bash
pip install -r requirements.txt
python -m heartsafe.run            # simulate, run agent + baseline, evaluate, equity audit -> data/
streamlit run app.py               # dashboard
pytest                             # tests
```

Plug in real model outputs (shared HeartSafe JSON format):

```bash
python -m heartsafe.run --imaging ../imaging/results/clean/exams --ecg ../ecg/results/exams
```

Use a real Medly-style export instead of the simulation:
`python -m heartsafe.run --feed export.csv --patients-csv patients.csv`
(columns: patient_id, day, submitted, weight_kg, sbp, hr, sob, swelling; patients:
patient_id, remoteness, sex, age, age_band, device, ecg_kit).

## What the agent does with each Medly alert

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
heartsafe/simulate.py  Medly-style cohort (+ scripted demo patient DEMO-MARY)
heartsafe/medly.py     Medly adapter and Medly-style alert rules
heartsafe/reliability.py  reading, ECG and MRI reliability checks
heartsafe/signals.py   ECG quality index; loaders for ECG / imaging pipeline outputs
heartsafe/review.py    explainable review (template; optional LLM rewording)
heartsafe/tracker.py   SQLite case tracker with audit log and permissions
heartsafe/clinicians.py  assumed clinician response model
heartsafe/agent.py     the agent and the Medly-alone baseline
heartsafe/equity.py    subgroup audit with bootstrap CIs and safeguard proposals
heartsafe/evaluate.py  HeartSafe vs Medly alone on ground truth
heartsafe/run.py       command line entry point
app.py                 Streamlit dashboard
```

## LLM (optional)

Set `ANTHROPIC_API_KEY` and press "Reword with LLM" on a case. The model gets
only the structured facts and is told not to add any; the template is used if the
call fails. Decisions never depend on the LLM.
