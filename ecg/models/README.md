# Trained ECG models

PyTorch state dicts of the 1D ResNet in `../scripts/model.py`, single-lead (lead I) input. Trained on SDRE (one
H200 GPU) with the settings in `../config.yaml`.

| File | Model | Training data | Validation AUROC |
|---|---|---|---|
| `member_0.pt` … `member_4.pt` | normal / abnormal ensemble (5 members) | PTB-XL folds 1-8 | 0.885–0.887 |
| `rhythm_member_0.pt` … `rhythm_member_2.pt` | normal / atrial fibrillation / other ensemble (3 members) | PTB-XL folds 1-8 + 60% of PhysioNet/CinC 2017 | macro 0.899–0.900 |

Per-member details: `train_log.json`, `rhythm_train_log.json`. Held-out results: [../README.md](../README.md)
and the dashboard's Model reliability page.

## Use them instead of training

The scripts skip any ensemble member that already exists, so with these files in place they only predict,
calibrate and evaluate.

On the cluster, from `ecg/` (paths from `slurm/env.sh`):

```bash
source slurm/env.sh
mkdir -p $ECG_WORK/models && cp models/*.pt models/*.json $ECG_WORK/models/
sbatch --account=<ACCOUNT> --partition=cpu slurm/01_prepare.sbatch          # once: PTB-XL and CinC 2017 arrays
sbatch --account=<ACCOUNT> --partition=<GPU> slurm/02_train_evaluate.sbatch # predicts, calibrates, evaluates
sbatch --account=<ACCOUNT> --partition=<GPU> slurm/03_rhythm.sbatch         # adds the named rhythm
```

On any machine with PyTorch, from `ecg/scripts/` (after `01_prepare.py`):

```bash
python 03_predict_reliability.py --data work/ --models ../models --out results/ --config ../config.yaml
python 05_rhythm.py --data work/ --models ../models --results results/ --config ../config.yaml --ptbxl <ptbxl folder>
```

SHA-256:

```
7e3a20dd2571dac9e0fa5683665592e4fdb1d7f52da6b7d4f8743fa37ef27d42  member_0.pt
2d0597f17bed429563e9a2b7d37f20154701fb9711bc8c4f1f6af04d28236a1d  member_1.pt
305b9b7a5ca25a8c423a1c669fd6842a0142524af65840a79487207fa3f909b1  member_2.pt
28277c9824373bf595d9486c59a8caded8894ff6e84f9a2572f66db7dce93a1c  member_3.pt
20bb6a193b41722823b01b17b9a9151a9a2fc739c2c1d1276307a0644478fd61  member_4.pt
4f7b88502ee80d915c65025f79ad550967dd255bfbd082d2ba40fb1d4f580936  rhythm_member_0.pt
3d4fc00a601131ac435d8c531ba34eb51fa4671aa7c14571922b043499eac781  rhythm_member_1.pt
1e6c7e706cc4ba613d869dc16c388f1b92155340f55d94b7d3f8fc6cf43ac0e9  rhythm_member_2.pt
```

Data: PTB-XL (CC BY 4.0) and PhysioNet/CinC Challenge 2017; cite both when you use these weights. Research use
only; not a medical device.
