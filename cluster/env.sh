# Cluster environment, sourced by run_sessions.sbatch (and handy interactively: `source cluster/env.sh`).
# Edit the three settings below once for your account on the cluster.

# 1. Software: load conda and activate the environment created from environment.yaml.
#    `module avail miniconda` / `module avail anaconda` lists the conda modules installed.
module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate spikes2lfp

# 2. Data: the AllenSDK cache directory (contains visual-behavior-neuropixels-0.4.0/).
#    Outputs are written to $SPIKES2LFP_DATA_ROOT/spikes2lfp_v2/.
export SPIKES2LFP_DATA_ROOT=/path/to/allen_cache

# 3. Optional: keep numpy/BLAS from starting more threads than the CPUs Slurm allocated.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
