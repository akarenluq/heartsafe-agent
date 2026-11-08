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
slurm/                           Slurm jobs for SDRE or Alliance clusters (pip, no conda)
wheelhouse/                      offline nnU-Net wheels for Python 3.11 (not in git; copy with the code)
```

Splits: the original M&Ms Training split (vendors A, B) trains nnU-Net.
Validation + Testing (vendors A–D) are held out, so vendors C and D are unseen
scanners. That is the device-shift case the demo needs.

## Running on SDRE (U of Alberta)

SDRE has no guaranteed internet or GitHub access, so the code, the offline
`wheelhouse/` and the data all go in through Globus. Use **Project** storage
(5 TB), not Home (25 GB): preprocessed data and checkpoints need tens of GB.
Account, partition and module names come from the SDRE Technical Wiki you
receive at onboarding.

1. **Transfer with Globus** (Globus Connect Personal on your PC, to RDSS, then into SDRE Project):
   `D:\WAI\archive.zip` and the whole `D:\WAI\HeartSafe_WAI-Hackathon` folder, wheelhouse included.
   Put both in one folder on Project storage, called `<PROJECT>/heartsafe` below.
2. **Unzip and check:**
   ```bash
   cd <PROJECT>/heartsafe
   unzip -l archive.zip | head -40      # confirm it is M&Ms first
   unzip -q archive.zip -d mnms
   cd HeartSafe_WAI-Hackathon/imaging   # submit every job from this folder
   ```
3. **Site settings, once:**
   ```bash
   bash slurm/configure_site.sh --root <PROJECT>/heartsafe \
        --account <ACCOUNT> --gpu-partition <AI/H200 PARTITION> --cpu-partition <CPU PARTITION> \
        --modules "StdEnv/2023 python/3.11"
   ```
   Re-run it any time; it rewrites the `#SBATCH` lines and `slurm/site.env`.
   GPUs are requested as `--gpus-per-node` (SDRE default form); pass `--gpus h200:1` to pin a type.
   Check the module names with `module avail python` and `module spider StdEnv`.
4. **Environment, once, on a login node:** `bash slurm/setup_env.sh`
   It installs torch from the site wheels and nnU-Net from `wheelhouse/` (no internet needed),
   and prints the torch and nnU-Net versions. If torch is missing from the site wheels,
   ask research.support@ualberta.ca which torch module or Apptainer image to use.
5. **Prepare (CPU):** `sbatch slurm/01_prepare.sbatch`
   Expect about 150 training subjects and 170 held-out (34 val + 136 test).
   If 00_inspect reports no `*_sa_gt.nii.gz`, stop: the archive is not M&Ms.
6. **Smoke test (GPU, ~30 min):** `sbatch slurm/02_smoke_test.sbatch`
   Continue only if the log ends with `SMOKE TEST PASSED`.
7. **Train:** `sbatch slurm/03_train.sbatch` (2d, fold 0, 250 epochs; an H200 should finish well inside the 10 h limit).
   On timeout: `sbatch --export=ALL,RESUME=1 slurm/03_train.sbatch`.
8. **Predict, reliability, audit, evaluation:** `sbatch slurm/04_predict_and_evaluate.sbatch`
   Outputs in `<PROJECT>/heartsafe/results/`:
   - `clean/exam_results.csv`, `clean/exams/*.json` (hand these to the agent), `clean/uncertainty_maps/`
   - `faulty/...` the same for fault-injected exams
   - `audit/subgroup_audit.csv`, `audit/forest_*.png`, `audit/safeguard_proposals.json`
   - `reliability.json` sensitivity/specificity vs baseline
9. **Getting results out:** follow SDRE's export rules (ask research.support@ualberta.ca if unsure).
   M&Ms is public and de-identified, and the results are summary numbers and JSON, but SDRE still controls export.

The same scripts run on an Alliance cluster: pass that cluster's account,
partition and storage path to `configure_site.sh`.

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
