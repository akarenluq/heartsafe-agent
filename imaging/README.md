# HeartSafe Agent: imaging path

Cardiac MRI segmentation on M&Ms with nnU-Net v2, plus the imaging half of the
agent's reliability check and the imaging equity audit. Each exam (ED + ES of
one subject) becomes one JSON in the shared interface:
`{prediction, probability, uncertainty, quality, shift_score, metadata}`
plus `flags` and a `suggested_action` (accept / repeat / escalate).

```
imaging/config.yaml                      thresholds and safeguards (the LLM never edits this)
scripts/00_inspect_dataset.py    check the download really is M&Ms
scripts/01_convert_mnms_to_nnunet.py   4D cines -> ED/ES 3D cases + metadata.csv
scripts/04_postprocess_reliability.py  volumes, EF, uncertainty maps, quality, shift, plausibility
scripts/05_inject_faults.py      faulty copies (noise, blur, slice drop, contrast) for evaluation
scripts/06_subgroup_audit.py     subgroup Dice / LVEF error with bootstrap 95% CIs, safeguard proposals
scripts/07_eval_reliability.py   agent vs no-agent baseline
slurm/                           env + jobs for an Alliance cluster (pip, no conda)
```

Splits: the original M&Ms Training split (vendors A, B) trains nnU-Net.
Validation + Testing (vendors A–D) are held out, so vendors C and D are unseen
scanners. That is the device-shift case the demo needs.

## Steps on the cluster

1. **Upload and unzip** (from your PC, then on the cluster):
   ```bash
   scp archive.zip yassmin2@rorqual.alliancecan.ca:~/scratch/heartsafe/
   # on the cluster
   cd ~/scratch/heartsafe && unzip -l archive.zip | head -40   # check before unzipping
   unzip -q archive.zip -d mnms
   ```
   Clone the repo into `~/scratch/heartsafe/` (`git clone https://github.com/yassmin2000/HeartSafe_WAI-Hackathon.git`); the code is then in `HeartSafe_WAI-Hackathon/imaging`.

2. **Environment, once, on a login node:** `bash slurm/setup_env.sh`

3. **Account:** replace `def-CHANGEME` in every `slurm/*.sbatch` with your allocation
   (`sed -i 's/def-CHANGEME/def-YOURPI/' slurm/*.sbatch`). Check the GPU line
   (`--gpus-per-node=h100:1`) against the cluster docs.

4. **Prepare (CPU):** `sbatch slurm/01_prepare.sbatch`
   The log should show about 150 training subjects and 170 held-out subjects (34 val + 136 test). If 00_inspect reports no `*_sa_gt.nii.gz`, stop: the archive is not M&Ms.

5. **Smoke test (GPU, ~30 min):** `sbatch slurm/02_smoke_test.sbatch`
   Runs 5 epochs, predicts 4 cases and postprocesses them. Continue only if the log ends with `SMOKE TEST PASSED`.

6. **Train:** `sbatch slurm/03_train.sbatch` (2d, fold 0, 250 epochs).
   On timeout: `sbatch --export=ALL,RESUME=1 slurm/03_train.sbatch`.

7. **Predict, reliability, audit, evaluation:** `sbatch slurm/04_predict_and_evaluate.sbatch`
   Outputs in `~/scratch/heartsafe/results/`:
   - `clean/exam_results.csv`, `clean/exams/*.json` (hand these to the agent), `clean/uncertainty_maps/`
   - `faulty/...` the same for fault-injected exams
   - `audit/subgroup_audit.csv`, `audit/forest_*.png`, `audit/safeguard_proposals.json`
   - `reliability.json` sensitivity/specificity vs baseline

## Thresholds

`auto` thresholds are calibrated once and saved in
`$RAW/reliability_thresholds.json`, so the clean and faulty runs use identical values:
- uncertainty: 90th percentile on the held-out validation exams
- scanner shift: 95th percentile of training-scan distances
- minimum SNR: 5th percentile of training scans

Delete that file to recalibrate after changing `config.yaml`.

## Safeguard loop (equity demo)

1. `06_subgroup_audit.py` writes proposals, all `approved: false`.
2. An administrator copies one into `safeguards:` in `config.yaml` and sets `approved: true`.
3. Re-run step 4 of the predict job (`04_postprocess_reliability.py`, same thresholds file).
   That subgroup is now escalated at a lower uncertainty threshold.
   `audit/safeguard_cost.json` counts the extra escalations.

## Notes

- Ground truth is stored under `_eval` in each JSON for evaluation only; the agent must not read it.
- The pipeline was tested end to end on synthetic M&Ms-style data, not yet on the real dataset.
