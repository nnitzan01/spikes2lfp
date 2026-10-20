# Cluster environment (NYU BigPurple), sourced by run_sessions.sbatch; also handy interactively:
#   source cluster/env.sh

module load anaconda3
eval "$(conda shell.bash hook)"          # makes `conda activate` work in batch jobs (no `conda init` needed)
conda activate spikes2lfp

# The pip build of PyTorch bundles its own CUDA runtime and cuDNN (8.7) and only needs the GPU driver.
# The cluster's cuda/11.8 module (loaded with anaconda3/gpu/new, or left over in the shell) puts
# cuDNN 8.2 from /gpfs/share/apps/cuda on LD_LIBRARY_PATH, which PyTorch would load instead and
# refuse to run with. Drop those entries.
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/apps/cuda/' | paste -sd: -)

# AllenSDK cache directory (contains visual-behavior-neuropixels-0.4.0/).
# Outputs are written to $SPIKES2LFP_DATA_ROOT/spikes2lfp_v2/ (first-version results stay in spikes2lfp/).
export SPIKES2LFP_DATA_ROOT=/gpfs/data/buzsakilab/nitzan01/Allen2022

# keep numpy/BLAS to the CPUs Slurm allocated
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
