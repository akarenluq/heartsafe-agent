# HeartSafe Agent — WAI Canada Hackathon 2026

**A safety and follow-up layer for remote heart-failure monitoring.** HeartSafe sits between an existing
remote-monitoring platform (our main reference is Medly, UHN, used in remote and Indigenous communities;
TELUS Home Health Monitoring and Cloud DX work the same way) and the clinical team. It checks whether each
alert rests on reliable data, explains it, tracks every escalation until a clinician closes it, follows the
instruction through to the patient, and watches whether some communities are served worse than others.

**The clinical team makes every decision.** HeartSafe checks, explains, reminds and follows up; it never
closes a case, changes a treatment or calls 911 on its own.

![A short tour of the HeartSafe dashboard](docs/demo/heartsafe-tour.gif)

> **Demo video (3 min, captioned):** [docs/demo/heartsafe-demo.mp4](docs/demo/heartsafe-demo.mp4).
> To play it inline here, see [Demo video](#demo-video). **Abstract:** [docs/HeartSafe_Agent_Abstract.pdf](docs/HeartSafe_Agent_Abstract.pdf).

<!-- DEMO VIDEO: drag docs/demo/heartsafe-demo.mp4 into this README in GitHub's web editor and paste the
     generated https://github.com/user-attachments/... link on the line below, then delete this comment. -->

## Contents

1. [Why HeartSafe](#why-heartsafe)
2. [How it fits together](#how-it-fits-together)
3. [Features](#features)
4. [Results](#results)
5. [Dashboard walkthrough](#dashboard-walkthrough)
6. [Demo video](#demo-video)
7. [Quick start (no training needed)](#quick-start-no-training-needed)
8. [Trained models](#trained-models)
9. [Retrain or re-run the models on a cluster](#retrain-or-re-run-the-models-on-a-cluster)
10. [Repository layout](#repository-layout)
11. [Limits](#limits)

## Why HeartSafe

After a heart-failure admission, remote monitoring sends daily weight, blood pressure, heart rate and
symptoms to a nurse. The platform raises alerts, but:

- **Many alerts rest on bad data:** a new scale, a reading in pounds, a loose ECG lead, a noisy recording.
- **Alerts can stall:** nobody tracks whether a case was acknowledged, acted on and closed.
- **The instruction can be lost:** "take an extra diuretic" is sent once; nobody checks it was done or worked.
- **Remote patients can wait longer,** and nobody measures it.

HeartSafe adds the missing checks and follow-up on top of the platform. It complements the platform and
scheduled follow-up visits; it does not replace them.

## How it fits together

```
Platform feed ──► HeartSafe agent ──────────────────────► clinical team ──► patient loop
(readings,        │ reliability checks (readings, ECG, MRI)   (acknowledge,      (confirm, act,
 symptoms,        │ named ECG rhythm, heart rate              review, decide)    readings checked)
 alerts)          │ explainable review + follow-up watch
                  │ sleep and breathing log
                  └ equity monitor ──► safeguards approved by an administrator
```

1. **Monitoring-platform feed.** Daily weight, blood pressure, heart rate, oxygen saturation, symptoms,
   diuretic taken and post-discharge labs, with the platform's own normal / caution / critical alert.
   The cohort is simulated (2,001 patients × 90 days), modelled on Medly, since these platforms are not open source.
2. **Model signals.** A single-lead ECG classifier and rhythm model (`ecg/`) and cardiac MRI segmentation
   (`imaging/`) each write one JSON per exam in a shared format:
   `{prediction, probability, uncertainty, quality, shift_score, metadata, flags, suggested_action}`.
3. **Reliability checks.** Each reading and model output ends in accept / repeat measurement / escalate.
4. **Explainable review.** Every escalation gets a structured review in plain language.
5. **Tracked follow-up.** Cases go created → acknowledged → reviewed → resolved, with deadlines, reminders,
   backup routing and an audit log. Then the **patient loop** follows the clinician's instruction to the patient.
6. **Equity monitor.** Subgroup measures with bootstrap 95% CIs; gaps become safeguard proposals.
7. **Sleep and breathing.** Patients log sleep and breathing exercises; worse sleep opens a routine check-in call.
8. **Evaluation.** HeartSafe against the platform alone on the same simulated patients, and each model
   against injected faults and an unseen device or scanner.

## Features

### Reliability checks on every alert
- **Implausible values and unit errors** (154 kg after 70 kg, a reading in pounds, a cuff artifact): ask for a repeat, no case.
- **New scale:** an overnight weight jump with no symptoms asks for a repeat; if it persists, a nurse calls to
  verify. A confirmed new scale re-baselines the readings; the same scale means the gain is real.
- **Alerts that no longer stand** on the corrected data are filtered and logged, never silently dropped.
- **Missed readings** for 3 days become an outreach task.

### ECG: is it reliable, and what does it show?
- **Every recording is checked** (weekly, on symptom days, and when the agent asks on an alert day).
  A recording that is not usable (disconnected, noisy, low quality, model uncertain, unlike the training
  data) gets "please record it again".
- **Normal or abnormal:** a single-lead ECG ensemble (PTB-XL lead I), calibrated, with quality and device-shift checks.
- **Rhythm named:** a second model gives normal / atrial fibrillation (AF) / other abnormal.
  - **New AF** is reported to the care team once: **critical with symptoms**, caution without (raised to
    critical if symptoms follow).
  - **Known AF** (in the enrollment record) is logged and never reported as new.
- **Heart rate from the R-peaks,** checked against the patient's own limits set at enrollment.
- **Emergency level:** chest pain or fainting **and** a reliable abnormal ECG at the same time. The patient is
  told at once "If you feel worse, call 911"; a clinician must start a video call within 15 minutes and records
  "call 911" or "cancel the alarm" with a reason.

### Cardiac MRI
- nnU-Net segmentation of the left and right ventricles and myocardium, with volumes and ejection fraction.
- Reliability checks: image quality, uncertainty maps, scanner shift, plausibility. A result flagged by the
  pipeline becomes an imaging review case.

### Explainable review (no LLM needed)
Why it was flagged, symptoms, recent trend, a plain-language summary of the ECG and MRI reports, data-quality
notes, sleep and breathing, the follow-up status and a suggested next step from the program protocol.
An LLM-written version (Claude, read-only tools) is optional future work, off by default.

### Tracked cases and follow-up watch
- Deadlines by severity (acknowledge within 15 min for an emergency, 4 h critical, 24 h caution, 72 h low).
- Missed deadlines send reminders; a second miss goes to a backup clinician.
- A follow-up watch re-checks every case and patient loop every 2 hours.
- **Only a clinician** can acknowledge, review or resolve. Every action is in an append-only audit log.

### Patient loop
After the clinician decides, HeartSafe sends the instruction and follows it: **confirmed → action done →
readings checked 24 h later**. If the patient does not confirm, it contacts their caregiver or the community
health worker. Readings that do not improve reopen the case. Only the patient side can confirm.

### Sleep tracker and breathing exercises
Poor sleep and shallow breathing are common in the weeks after a heart-failure admission or heart surgery.
- **Patient page:** last night's hours, times woken, "slept propped up", "woke up short of breath", what kept
  them awake, and a chart against their own usual night.
- **Breathing exercises:** a paced timer (4 s in / 6 s out, or with a 2 s pause), 2 or 5 minutes, counted
  against the care team's plan (3 sessions a day by default). It gives no medical advice.
- **Sleep rule:** compared with the patient's own usual, not a fixed number of hours. Worse sleep opens a
  **low-priority check-in call** for the nurse (at most one in 21 days), never a clinical alert on its own.
  Breathlessness at night is passed on at caution level, as the platform already does.

### Equity monitor
Compares groups (remoteness, sex, age, device) on observable measures with bootstrap 95% CIs. A gap is flagged
only when a group's whole interval is worse than overall. Gaps become **safeguard proposals** with their workload
cost (for example a tighter deadline, or contacting helpers earlier); an administrator approves them, and the
cohort is re-run to show what changed.

## Results

Simulated cohort, 2,001 patients × 90 days. Both arms use the same patients, nurses and response times; the only
difference is whether HeartSafe sits between the platform and the care team.

| Measure | HeartSafe | Platform alone |
|---|---|---|
| Median hours from first alert to acknowledgement (true heart-failure episodes) | **6.7** | 12.8 |
| Acknowledged within 24 h (true episodes) | **80%** | 70% |
| True episodes escalated | 100% | 100% |
| Alerts caused by faulty readings that reached the nurse | **14%** | 100% |
| False alarms per 100 patient-months | **60** | 75 |
| Patients who completed the clinician's instruction | **96%** | 76% |
| New-AF spells reported to the care team (median 2 days from onset) | 94% | — |
| Sleep check-in calls per 100 patient-months (routine, low priority) | 5.3 | — |

Models, on held-out data:

| Model | Result |
|---|---|
| ECG normal / abnormal (PTB-XL lead I) | AUROC 0.887; on a new device (AliveCor, PhysioNet/CinC 2017) 0.672, which is why device shift is checked |
| ECG reliability check | 100% of injected faults caught; 81% of clean ECGs accepted |
| ECG rhythm, AF | PTB-XL sensitivity 0.86, specificity 0.99; new device 0.76 / 0.96 |
| Heart rate from R-peaks | sinus tachycardia caught 89%, sinus bradycardia 63% |
| Cardiac MRI segmentation (M&Ms, 170 subjects) | LV Dice 0.89–0.94 on every vendor, including two never seen in training; LVEF error 3.6–5.3 points |
| MRI reliability policy | 92% of injected faults caught; wrong predictions missed: 9, against 60 with no check |

Response times, device fault rates, sleep and exercise habits are simulation assumptions (`agent/config.yaml`),
identical for both arms. The simulation does not model the effect of treatment. Details:
[agent/README.md](agent/README.md).

## Dashboard walkthrough

Start the dashboard (see [Quick start](#quick-start-no-training-needed)) and follow these steps. Each step says
what to click and what to point out; the screenshots come from the recorded demo. Choose the role in the
sidebar: **Clinician** and **Administrator** see the care-team pages, **Patient** sees only the patient's own page.

### Care team (sign in as Clinician)

**1. Clinician queue.** The landing page: open cases, most urgent first, with the follow-up watch that
re-checks every case and patient loop every 2 hours.
*Point out:* the four counters, the overdue items the watch found, and the severity and type filters.

![Clinician queue](docs/demo/screenshots/01-clinician-queue.png)

**2. One case: the explainable review.** Select a row (the first case opens by default). The review says why it
was flagged, the symptoms, recent readings, the ECG and MRI reports in plain words, and the suggested next step.
The chart on the right switches between weight, blood pressure, heart rate and sleep.

![Case review](docs/demo/screenshots/02-case-review.png)

**3. Follow-up and audit log.** Open the **Follow-up** tab for deadlines, reminders and the patient loop
(instruction sent → confirmed → action done → readings checked). Open **Audit log** for every action, who did
it and when. At the bottom of the case, the clinician acknowledges, reviews or resolves it (a resolution needs a
note). Switch to Administrator to show that the button is locked.

![Case follow-up](docs/demo/screenshots/03-case-followup.png)
![Audit log](docs/demo/screenshots/04-case-audit.png)

**4. Patient journey.** Open **Patient journey**; the demo patient (68, remote community) is selected. The chart
shows her daily readings with a dot on every day HeartSafe acted.
*Tell her story from the table:* an abnormal ECG on day 0 added to the chart; a sleep check-in on day 10; a weight
jump on days 20–23 that turned out to be a new scale (a nurse verified it, so no false alarm); unusable ECGs
re-recorded; a real escalation on day 50 (breathless lying flat) that went through the patient loop and was
reopened when her readings did not improve.

![Patient journey](docs/demo/screenshots/06-patient-journey.png)
![Alerts and HeartSafe actions](docs/demo/screenshots/07-journey-actions.png)
![Cases for this patient](docs/demo/screenshots/08-journey-cases.png)

**5. Equity audit.** Open **Equity audit**. Choose a measure: each group's average with its 95% CI against the
overall average; groups whose whole interval is worse are marked "gap". The **Safeguards** tab turns gaps into
proposals with their workload cost; sign in as **Administrator** to approve one and press "Apply approved
safeguards now" to re-run the cohort (about 6 minutes).

![Equity audit](docs/demo/screenshots/09-equity-gaps.png)
![Safeguards](docs/demo/screenshots/10-equity-safeguards.png)

**6. Results.** Open **Results**: HeartSafe against the platform alone. Then the tabs: **Patient loop**,
**Heart rhythm** (new AF reported, known AF logged, the false new-AF reports, the two questions for the
cardiologists) and **Sleep and breathing** (check-in workload, breathing plan adherence).

![Results](docs/demo/screenshots/11-results.png)
![Patient loop results](docs/demo/screenshots/12-results-patient-loop.png)
![Heart rhythm results](docs/demo/screenshots/13-results-rhythm.png)
![Sleep and breathing results](docs/demo/screenshots/14-results-sleep.png)

**7. Model reliability.** Open **Model reliability**: the ECG classifier on held-out data and on a new device,
faults caught by type, the rhythm model and heart rate; then the **Cardiac MRI** tab with Dice by scanner vendor
and the reliability policy comparison.

![ECG model](docs/demo/screenshots/15-models-ecg.png)
![Rhythm model](docs/demo/screenshots/16-models-rhythm.png)
![MRI model](docs/demo/screenshots/17-models-mri.png)

**8. How it works.** The seven steps on one page, useful as a closing slide.

![How it works](docs/demo/screenshots/18-how-it-works.png)

### Patient (sign in as Patient)

**9. Sleep.** Choose **Patient** in the sidebar (the demo patient is selected). Her page shows last night, her
usual night and breathing sessions today. *Demo:* enter 4 hours, woken 3 times, tick "I woke up short of breath",
and save. Because breathlessness at night is a symptom the platform alerts on, the care team is told and a nurse
is asked to call; the message ends with "If you feel worse, call 911."

![Patient sleep page](docs/demo/screenshots/19-patient-sleep.png)
![Sleep saved](docs/demo/screenshots/20-patient-sleep-saved.png)

**10. Breathing exercises.** Open **Breathing exercises**, choose an exercise and a length, and press **Start**:
the circle grows on the breath in and shrinks on the breath out. Press **I finished this session** to log it.
**My entries** lists everything she logged.

![Breathing exercises](docs/demo/screenshots/21-breathing.png)
![Breathing timer](docs/demo/screenshots/22-breathing-timer.png)
![Session logged](docs/demo/screenshots/23-breathing-logged.png)
![My entries](docs/demo/screenshots/24-my-entries.png)

**11. Back to the care team.** Switch to **Clinician**, open **Patient journey**: her sleep and breathing entries
are next to her other readings, and in every case review.

![The care team sees the entries](docs/demo/screenshots/25-clinician-sees-sleep.png)

The demo clock stands at day 90, so a patient can log one night in the app; the earlier nights come from the
platform feed. Entries are kept in `agent/data/selfcare.db`.

## Demo video

A captioned recording of the walkthrough above is in
[docs/demo/heartsafe-demo.mp4](docs/demo/heartsafe-demo.mp4) (1440 × 900, 3 minutes, no sound). It plays in
PowerPoint (Insert → Video → This Device) and any browser.

**Play it inline in this README:** open README.md on GitHub, click the pencil to edit, drag the mp4 into the
editor where the `DEMO VIDEO` comment is, and commit. GitHub uploads it and inserts a
`https://github.com/user-attachments/...` link that plays in the page (files up to 100 MB).

**Re-record it** after changing the dashboard. The recorder runs the dashboard on a copy of the data, so the demo
changes nothing:

```bash
cd agent
cp -r data ../docs/demo/demo_data && rm -f ../docs/demo/demo_data/selfcare.db
streamlit run ../docs/demo/demo_app.py --server.port 8503 -- ../docs/demo/demo_data
# in a second terminal:
pip install playwright imageio-ffmpeg && python -m playwright install chromium-headless-shell
python ../docs/demo/record_demo.py http://localhost:8503 ../docs/demo/new_recording
```

It writes one screenshot per step and `heartsafe-demo.mp4`. The captions and the order of steps are in
`docs/demo/record_demo.py`.

**Record it yourself with narration** (for the presentation):

1. Start the dashboard, open http://localhost:8501 in a full-screen browser window at 100% zoom, and close other tabs.
2. Record the screen with one of:
   - **Windows:** Xbox Game Bar, `Win + Alt + R` to start and stop (saved in Videos → Captures), or the Snipping Tool's video mode;
   - **PowerPoint:** Insert → Screen Recording, which puts the video straight on a slide;
   - **macOS:** `Cmd + Shift + 5`;
   - **OBS Studio** for a webcam overlay.
3. Follow the steps of the walkthrough. A three-minute script:

| Time | Page | What to say |
|---|---|---|
| 0:00 | Clinician queue | "Remote monitoring raises alerts; HeartSafe makes sure each one rests on good data and is followed until a clinician closes it." |
| 0:25 | Case review | "Every case explains itself: why, symptoms, ECG and MRI in plain words, and a next step. The clinician decides." |
| 0:50 | Follow-up, audit log | "Deadlines, reminders, backup routing, and an audit log of every action." |
| 1:10 | Patient journey | "Her weight jumped, but a nurse verified a new scale, so no false alarm. When she was really breathless, the case went through the patient loop and reopened when she did not improve." |
| 1:45 | Results, Heart rhythm | "Acknowledged in 6.7 hours instead of 12.8, 86% of faulty-reading alerts stopped. New AF is reported; known AF is not." |
| 2:15 | Equity audit | "Are remote patients served worse? Gaps become safeguards an administrator approves." |
| 2:35 | Patient page | "Patients log sleep and do breathing exercises. Worse sleep starts a routine call; breathlessness at night reaches the care team." |
| 3:00 | How it works | "HeartSafe checks, explains and follows up. The clinical team makes every decision." |

## Quick start (no training needed)

You need Python 3.11. Nothing has to be trained to run the agent and the dashboard.

```bash
git clone https://github.com/yassmin2000/HeartSafe_WAI-Hackathon.git
cd HeartSafe_WAI-Hackathon/agent
python -m venv .venv && .venv\Scripts\activate      # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m heartsafe.run        # simulate the cohort, run HeartSafe and the platform alone (about 6 min)
streamlit run app.py           # dashboard at http://localhost:8501
pytest                         # tests
```

- **With the real model outputs** (`ecg/results/`, `imaging/results/`), the run uses each pipeline's calibrated
  decision on every exam, and the Model reliability page shows the held-out results. These folders are not in
  git (the MRI outputs are 3.3 GB); copy them with `bash fetch_results.sh` from the cluster.
- **Without them,** the run uses simulated model outputs with the same error rates (`--simulated-models`), and
  everything else works the same.
- **A real platform export** instead of the simulation: `python -m heartsafe.run --feed export.csv --patients-csv patients.csv`
  (columns in [agent/README.md](agent/README.md)).

## Trained models

The trained weights are kept so the models do not have to be trained again.

| Model | Where | Size | Use |
|---|---|---|---|
| ECG normal / abnormal, 5-member ensemble | `ecg/models/member_*.pt` | 5 × 2 MB | `03_predict_reliability.py --models` |
| ECG rhythm (normal / AF / other), 3-member ensemble | `ecg/models/rhythm_member_*.pt` | 3 × 2 MB | `05_rhythm.py --models` |
| Cardiac MRI nnU-Net v2, 2d, fold 0, 250 epochs | `imaging/models/heartsafe-mri-nnunet-2d.zip` (Git LFS) | 171 MB | `nnUNetv2_predict` |

Training logs are next to the ECG weights (`train_log.json`, `rhythm_train_log.json`). Details and the exact
commands are in [ecg/models/README.md](ecg/models/README.md) and [imaging/models/README.md](imaging/models/README.md).

- **ECG:** prepare the data once (`ecg/slurm/01_prepare.sbatch`, PTB-XL and PhysioNet/CinC 2017), copy
  `ecg/models/*` into the work folder's `models/`, then submit `02_train_evaluate.sbatch` and `03_rhythm.sbatch`
  as usual. Both skip any ensemble member that already exists, so they only predict, calibrate and evaluate
  (a few minutes on one GPU).
- **MRI:** the zip is stored with Git LFS, so install Git LFS before cloning (or run `git lfs pull` afterwards).
  The checkpoint is an inference-only copy (network weights in half precision, no optimizer state; the full
  training checkpoint is 268 MB). On three held-out test cases it gave voxel-for-voxel the same segmentation as
  the full checkpoint. Unzip it into `$nnUNet_results` and run `imaging/slurm/04_predict_and_evaluate.sbatch`;
  training is not needed.

## Retrain or re-run the models on a cluster

On a Slurm cluster (SDRE or Alliance):

1. **MRI:** follow [imaging/README.md](imaging/README.md) (prepare, smoke test, train 250 epochs ≈ 1 h on an
   H200, predict and evaluate).
2. **ECG**, from `ecg/`: `slurm/01_prepare.sbatch`, then `slurm/02_train_evaluate.sbatch`, then
   `slurm/03_rhythm.sbatch` for the rhythm model (about 3 minutes each on one GPU).
3. **Copy the results to this machine:** `bash fetch_results.sh` (Git Bash; needs an open SSH connection, see
   the top of the script). Results land in `ecg/results/` and `imaging/results/`.
4. Re-run the agent: `cd agent && python -m heartsafe.run`.

## Repository layout

| Folder | What |
| --- | --- |
| `agent/` | Platform adapter, reliability layer, rhythm and heart-rate findings, reviews, case tracker, patient loop, sleep and breathing log, equity monitor, evaluation, dashboard ([README](agent/README.md)) |
| `ecg/` | Single-lead ECG classifier (PTB-XL lead I), ECG reliability, PhysioNet/CinC 2017 device-shift test, rhythm model; trained weights in `ecg/models/` |
| `imaging/` | Cardiac MRI segmentation (nnU-Net, M&Ms), volumes and EF, imaging reliability, imaging equity audit ([README](imaging/README.md)) |
| `docs/` | The project abstract ([PDF](docs/HeartSafe_Agent_Abstract.pdf), source in `docs/abstract/`); in `docs/demo/`, the screenshots, demo video and GIF, and the scripts that record them |
| `course/` | An interactive course that explains the codebase (open `course/index.html`) |
| `plans/` | Execution plan |

Data and model outputs are never committed (see `.gitignore`); the trained weights are (the MRI zip through Git LFS).

## Limits

- **Simulated cohort.** Patients, response times, device faults, sleep and exercise habits are simulation
  assumptions; the results compare workflows, not clinical outcomes.
- **Single-lead ECG.** 20 of 49 simulated new-AF reports were on recordings whose true rhythm was not AF; every
  report asks the clinician to confirm on a 12-lead ECG.
- **Not a medical device.** HeartSafe has not been validated clinically. It supports the clinical team's
  process and never replaces their judgement.
- **Data terms.** The models were trained on public, de-identified datasets (PTB-XL, PhysioNet/CinC 2017, M&Ms);
  follow each dataset's terms when sharing the weights or outputs.
