# Sourced by every job. Edit HS_ROOT if you keep the project elsewhere.
module load StdEnv/2023 python/3.11
export HS_ROOT=$HOME/scratch/heartsafe
export HS_CODE=$HS_ROOT/HeartSafe_WAI-Hackathon/imaging
source $HOME/envs/heartsafe/bin/activate

export nnUNet_raw=$HS_ROOT/nnUNet_raw
export nnUNet_preprocessed=$HS_ROOT/nnUNet_preprocessed
export nnUNet_results=$HS_ROOT/nnUNet_results
export nnUNet_compile=f          # avoids torch.compile issues on cluster nodes
export nnUNet_n_proc_DA=${SLURM_CPUS_PER_TASK:-8}
export RAW=$nnUNet_raw/Dataset201_HeartSafeMNMs
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results $HS_ROOT/logs
