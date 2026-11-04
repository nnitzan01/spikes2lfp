#!/bin/bash
# Run one experiment (experiments/<name>.py) on one session. Submit from the repository root:
#
#   EXPERIMENT=stimulus_clock sbatch --array=1-89%8 cluster/run_experiment.sbatch
#   EXPERIMENT=baselines sbatch --array=1-89%8 cluster/run_experiment.sbatch
#   EXPERIMENT=baselines EXTRA_ARGS="--lstm none" sbatch --array=1 cluster/run_experiment.sbatch
#   EXPERIMENT=ranked_ablation sbatch --array=1-89%8 cluster/run_experiment.sbatch
#   EXPERIMENT=region_ablation sbatch --array=1-89%8 cluster/run_experiment.sbatch
#   EXPERIMENT=spectral_r2 sbatch --array=1-89%8 cluster/run_experiment.sbatch
#
# An array task runs line N of cluster/sessions.txt. The experiments need the session's 'train'
# stage (run_sessions.sbatch); see each script's docstring for what else and what it saves.

#SBATCH --job-name=experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_short,gl40s_short,gpu4_medium,gpu8_medium
#SBATCH --gres=gpu:1
# nodes with an unusable GPU (see run_sessions.sbatch)
#SBATCH --exclude=a100-4012,a100-4033,gn-0020,gl40s-8009
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh

if [[ -z "${EXPERIMENT:-}" ]]; then
    echo "set EXPERIMENT to the name of a script in experiments/, e.g. EXPERIMENT=stimulus_clock"
    exit 2
fi
SESSION_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/sessions.txt)
SESSION_DIR="${SPIKES2LFP_DATA_ROOT}/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/${SESSION_ID}"

echo "job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID}) on $(hostname): ${EXPERIMENT} ${SESSION_DIR} ${EXTRA_ARGS:-}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && ! python -c "import torch; torch.zeros(1, device='cuda')" 2>/dev/null; then
    echo "GPU UNUSABLE on node $(hostname); resubmit with --exclude=$(hostname)"
    exit 3
fi

python -m "experiments.${EXPERIMENT}" "${SESSION_DIR}" ${EXTRA_ARGS:-}

sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State
