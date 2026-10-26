# Running the pipeline on NYU BigPurple (Slurm)

One array task per session: task N runs `run_session.py` on line N of `sessions.txt` (the 89 sessions
of the manuscript, i.e. those with CSD-derived L4 borders in `tables/layer_4_borders.csv`).

## One-time setup

```bash
cd /gpfs/home/nitzan01/Downloads/spikes2lfp
git fetch && git checkout revision/history-aware-models && git pull

module load anaconda3
conda env create -f environment.yaml        # new environment 'spikes2lfp' (or: conda env update -f environment.yaml)
```

`cluster/env.sh` loads `cuda/11.8` and `anaconda3`, activates the environment and sets
`SPIKES2LFP_DATA_ROOT=/gpfs/data/buzsakilab/nitzan01/Allen2022`; edit it if any of these change.
Submit every job from the repository root.

## Before the first submission

```bash
source cluster/env.sh
pytest                                               # ~3 min on CPU; synthetic data only

# pre-flight on a GPU node: software, the GPU code path, and every session's data
srun --gres=gpu:1 --partition=a100_short,gl40s_short,gpu4_medium --mem=16G --time=00:15:00 \
     bash -c "source cluster/env.sh && python cluster/check_setup.py"
```

Fix every `FAIL` line before going on.

Jobs run on any free GPU (A100, L40S or V100) in the partitions listed in `run_sessions.sbatch`;
`squeue -u $USER -o "%.12i %.12P %.8T %.20r"` shows the partition and why a job is still pending.

## Timing run (one session)

```bash
mkdir -p logs
sbatch --array=1 cluster/run_sessions.sbatch
tail -f logs/spikes2lfp_<jobid>.out
```

The log has one line per trained model (test R2, best epoch, seconds) and per attributed model
(number of windows, seconds, completeness error), and the duration of each stage; the job ends by
printing its wall time and peak memory (sacct). Set `--time` and `--mem` in `run_sessions.sbatch`
from these, with some margin. If attribution is too slow, reduce
`ATTR_SAMPLED`, `ATTR_EVENTS` or `PERI_EVENTS` in `config.py` (the number of windows per model is
3 x ATTR_SAMPLED + 4 x ATTR_EVENTS + PERI_EVENTS x 63).

## All sessions

```bash
sbatch --array=1-89%8 cluster/run_sessions.sbatch   # at most 8 tasks at once
squeue -u $USER
```

A task that fails or runs out of time can be resubmitted with the same index
(`sbatch --array=5,17 cluster/run_sessions.sbatch`): finished models and attribution files are kept
and only the missing ones are computed. Don't run two jobs on the same session at the same time.
`scancel <jobid>` cancels the whole array; `sacct -j <jobid> -X -o JobID,State,Elapsed` lists the tasks.

The script also takes a session directory, as the previous per-session scripts did:

```bash
sbatch cluster/run_sessions.sbatch /gpfs/data/buzsakilab/nitzan01/Allen2022/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/1044385384
```

Other runs:

```bash
STAGES=train,transfer sbatch --array=1-89%8 cluster/run_sessions.sbatch      # models only
EXTRA_ARGS="--seeds 3" STAGES=train sbatch --array=1-10 cluster/run_sessions.sbatch   # seed variability
```

Outputs: `$SPIKES2LFP_DATA_ROOT/spikes2lfp_v2/{models,variables,attrs,plots}/<session_id>/`
(the first, published model version stays in `spikes2lfp/`).

## Checking the outputs and pooling sessions

```bash
source cluster/env.sh
python -m experiments.pool_sessions check
```

lists every session with missing or stale outputs (models, attributions, summaries, unit ids, ...)
and prints the `sbatch` commands that redo them; GPU stages go to the usual partitions, the
CPU-only stages (`variables`, `summaries`, `peri`, `ipd`) to a CPU partition with `--gres=none`
(check the partition name with `sinfo -s`). Resubmitting is safe: finished outputs are kept.

When every session is complete, pool them on a CPU node (reads every attribution file; ~1 h):

```bash
srun --partition=cpu_short --mem=64G --time=04:00:00 bash -c "source cluster/env.sh && python -m experiments.pool_sessions pool"
```

Pooled arrays (channels aligned on the upper L4 border, one row per unit or per session) go to
`$SPIKES2LFP_DATA_ROOT/spikes2lfp_v2/pooled/`; `python -m experiments.pool_sessions --help`
describes each file. Only complete sessions are pooled.
