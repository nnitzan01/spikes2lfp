#!/bin/bash
# Run run_session.py on one session. Submit from the repository root:
#
#   mkdir -p logs
#   sbatch --array=1 cluster/run_sessions.sbatch                  # session on line 1 of sessions.txt (timing run)
#   sbatch --array=1-89%8 cluster/run_sessions.sbatch             # all 89 sessions, at most 8 at once
#   sbatch --array=5,17 cluster/run_sessions.sbatch               # resubmit tasks 5 and 17
#   sbatch cluster/run_sessions.sbatch <session_dir>              # one session given by its directory
#
#   STAGES=train,transfer sbatch --array=1-89%8 cluster/run_sessions.sbatch     # a subset of stages
#   EXTRA_ARGS="--seeds 3" sbatch --array=1-10 cluster/run_sessions.sbatch
#
# An array task runs line N of cluster/sessions.txt. A session that fails or runs out of time can be
# resubmitted: finished models and attribution files are kept and only the missing ones are computed.
# Don't run two jobs on the same session at the same time.

#SBATCH --job-name=spikes2lfp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh

if [[ $# -ge 1 ]]; then
    SESSION_DIR=${1%/}
else
    SESSION_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/sessions.txt)
    SESSION_DIR="${SPIKES2LFP_DATA_ROOT}/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/${SESSION_ID}"
fi
STAGES=${STAGES:-train,transfer,variables,attr,summaries,peri,ipd}

echo "job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-none}) on $(hostname)"
echo "session ${SESSION_DIR}, stages ${STAGES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python run_session.py "${SESSION_DIR}" --stages "${STAGES}" ${EXTRA_ARGS:-}

# wall time and peak memory, for setting --time and --mem
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State
