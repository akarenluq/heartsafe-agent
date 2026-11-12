# HeartSafe Agent — WAI Canada Hackathon 2026

Equity-aware trust and coordination layer on top of existing remote cardiac monitoring platforms
(main reference: Medly, UHN, used in remote and Indigenous communities; also TELUS Home Health Monitoring, Cloud DX): reliability checks,
explainable reviews, tracked follow-up and equity monitoring, so existing monitoring is safer and accountable.
The clinical team makes every decision.

## How it fits together

1. **Monitoring-platform feed.** Daily weight, blood pressure, heart rate, oxygen saturation, symptoms, diuretic taken and post-discharge labs for screened patients (inclusion criteria reviewed with cardiologists), with the platform's own
   normal / caution / critical alert (simulated cohort modelled on Medly, since these platforms are not open source).
2. **Model signals.** A single-lead ECG classifier (`ecg/`) and cardiac MRI segmentation (`imaging/`) each write
   one JSON per exam in a shared format:
   `{prediction, probability, uncertainty, quality, shift_score, metadata, flags, suggested_action}`.
   A second ECG model names the rhythm (normal / atrial fibrillation / other abnormal), and the heart rate is
   read from the R-peaks. New atrial fibrillation with symptoms is critical; atrial fibrillation already in the
   enrollment record is logged, not reported as new.
3. **Reliability checks.** Each input and model output is checked (data quality, signal or image quality,
   uncertainty, device or scanner shift, plausibility) and ends in accept / repeat measurement / escalate.
4. **Explainable review.** Every escalation gets a structured template review with plain-language ECG and
   MRI report summaries and its follow-up status, which a follow-up watch re-checks every 2 hours. An
   LLM-written version is optional future work.
5. **Tracked follow-up.** Cases go created → acknowledged → reviewed → resolved, with deadlines, reminders and an
   audit log. Only a clinician moves or closes a case. Then the **patient loop** follows the clinician's
   instruction to the patient: confirmed → action done → readings checked, contacting a caregiver or
   community health worker when the patient does not respond, and reopening when readings do not improve.
6. **Equity monitor.** Subgroup metrics with bootstrap 95% CIs; gaps become safeguard proposals that an
   administrator approves.
7. **Sleep and breathing exercises.** Patients log last night's sleep and their breathing-exercise sessions
   (paced timer, plan set by the care team). Sleep worse than the patient's own usual opens a low-priority
   check-in call for the nurse; it is never a clinical alert on its own.
8. **Evaluation.** HeartSafe against the monitoring platform alone on the simulated cohort, and each model against injected faults
   and an unseen device or scanner.

| Folder | What |
| --- | --- |
| `imaging/` | Cardiac MRI segmentation (nnU-Net, M&Ms), volumes and EF, imaging reliability, imaging equity audit |
| `ecg/` | Single-lead ECG classifier (PTB-XL lead I), ECG reliability, PhysioNet 2017 device-shift test, rhythm model (normal / AF / other) |
| `agent/` | Platform adapter, reliability layer, reviews, case tracker, sleep and breathing log, equity monitor, evaluation, dashboard |
| `plans/` | Execution plan |

Data, results and model weights are never committed (see `.gitignore`).

## Run it

1. **Train and evaluate the models** on a Slurm cluster (SDRE or Alliance): follow `imaging/README.md`, then
   submit the ECG jobs from `ecg/` (`slurm/01_prepare.sbatch`, then `slurm/02_train_evaluate.sbatch`, then
   `slurm/03_rhythm.sbatch` for the rhythm model).
2. **Copy the results to this machine:** `bash fetch_results.sh` (Git Bash; needs an open SSH connection, see the
   top of the script). Results land in `imaging/results/` and `ecg/results/`.
3. **Run the agent** on the simulated cohort with the real model outputs:
   `cd agent && pip install -r requirements.txt && python -m heartsafe.run`
4. **Open the dashboard:** `streamlit run app.py`
5. **Run the tests:** `pytest`
