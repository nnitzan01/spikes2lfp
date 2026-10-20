# Cluster environment (NYU BigPurple), sourced by run_sessions.sbatch; also handy interactively:
#   source cluster/env.sh

# No `module load cuda`: the pip build of PyTorch bundles its own CUDA runtime and cuDNN and only
# needs the GPU driver. The cuda/11.8 module puts an older cuDNN (8.2) on LD_LIBRARY_PATH, which
# PyTorch then loads instead of its own (8.7) and refuses to run.
module load anaconda3
eval "$(conda shell.bash hook)"          # makes `conda activate` work in batch jobs (no `conda init` needed)
conda activate spikes2lfp

# AllenSDK cache directory (contains visual-behavior-neuropixels-0.4.0/).
# Outputs are written to $SPIKES2LFP_DATA_ROOT/spikes2lfp_v2/ (first-version results stay in spikes2lfp/).
export SPIKES2LFP_DATA_ROOT=/gpfs/data/buzsakilab/nitzan01/Allen2022

# keep numpy/BLAS to the CPUs Slurm allocated
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
