#!/bin/bash
# Run run_session.py on one session per Slurm array task. Task N runs line N of cluster/sessions.txt.
# Submit from the repository root (outputs and logs are relative to it):
#
#   mkdir -p logs
#   sbatch --array=1 cluster/run_sessions.sbatch                          # first session only (timing run)
#   sbatch --array=1-89%8 cluster/run_sessions.sbatch                     # all sessions, at most 8 at once
#   STAGES=train,transfer sbatch --array=1-89%8 cluster/run_sessions.sbatch   # a subset of stages
#   EXTRA_ARGS="--seeds 3" sbatch --array=1-5 cluster/run_sessions.sbatch
#
# A task that runs out of time can simply be resubmitted with the same array index: finished models
# and attribution files are kept and only the missing ones are computed.
#
# The partition, time and memory below are starting points: check `sinfo -s` for the GPU partitions
# and their time limits, and adjust after the timing run (`sacct -j <jobid> -o JobID,Elapsed,MaxRSS`).

#SBATCH --job-name=spikes2lfp
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh

SESSION_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/sessions.txt)
SESSION_DIR="${SPIKES2LFP_DATA_ROOT}/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/${SESSION_ID}"
STAGES=${STAGES:-train,transfer,variables,attr,summaries,peri,ipd}

echo "task ${SLURM_ARRAY_TASK_ID}: session ${SESSION_ID} on $(hostname), stages ${STAGES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python run_session.py "${SESSION_DIR}" --stages "${STAGES}" ${EXTRA_ARGS:-}
