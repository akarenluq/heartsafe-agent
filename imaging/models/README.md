# Trained cardiac MRI model

nnU-Net v2, dataset `Dataset201_HeartSafeMNMs`, configuration `2d`, fold 0, trainer `nnUNetTrainer_250epochs`.
Trained on SDRE (one H200, about 1 hour) on the M&Ms training split (vendors A and B). Labels: 1 LV, 2 myocardium,
3 RV.

**File:** `heartsafe-mri-nnunet-2d.zip` in this folder (171 MB), stored with **Git LFS** because GitHub refuses
normal git files over 100 MB. Install Git LFS (`git lfs install`, included with Git for Windows) before cloning;
in a clone made without it the zip is a small text pointer, and `git lfs pull` downloads the real file.

SHA-256 of the zip: `a73c31f431fa9d978e9d67370610838a97846c3b6238bbdb5cb4cef7417c79cb`

## What is in the zip

```
Dataset201_HeartSafeMNMs/nnUNetTrainer_250epochs__nnUNetPlans__2d/
  dataset.json  dataset_fingerprint.json  plans.json
  fold_0/checkpoint_final.pth
```

`checkpoint_final.pth` is an **inference-only copy**: the network weights in half precision, without the
optimizer state, so it is 185 MB instead of 268 MB. On three held-out test cases it gave voxel-for-voxel the same
segmentation as the full training checkpoint. It cannot be used to resume training.

## Use it instead of training

```bash
unzip heartsafe-mri-nnunet-2d.zip -d $nnUNet_results
nnUNetv2_predict -i <folder of *_0000.nii.gz> -o <output folder> -d 201 -c 2d -f 0 \
    -tr nnUNetTrainer_250epochs --save_probabilities          # add -device cpu without a GPU
```

On the cluster, unzip it into `$nnUNet_results` (set by `imaging/slurm/env.sh`), skip the train step and submit
`slurm/04_predict_and_evaluate.sbatch`, which predicts the held-out and fault-injected exams and runs the
reliability checks and the audit.

Data: M&Ms (Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation challenge); follow its data terms
when sharing the weights. Research use only; not a medical device.
