# Running the pipeline on a Slurm cluster (NYU BigPurple)

One array task per session: task N runs `run_session.py` on line N of `sessions.txt` (the 89 sessions
of the manuscript, i.e. those with CSD-derived L4 borders in `tables/layer_4_borders.csv`).

## One-time setup

```bash
git clone https://github.com/nnitzan01/spikes2lfp.git && cd spikes2lfp
git checkout revision/history-aware-models          # or: git pull, in an existing clone

module load miniconda3                               # `module avail` for the exact name
conda env create -f environment.yaml                 # creates the 'spikes2lfp' environment
```

Edit `cluster/env.sh`: the conda module name and `SPIKES2LFP_DATA_ROOT` (the AllenSDK cache
directory that contains `visual-behavior-neuropixels-0.4.0/`).

## Before the first submission

```bash
source cluster/env.sh
pytest                                               # ~3 min on CPU; synthetic data only

# pre-flight on a GPU node: software, the GPU code path, and every session's data
srun --partition=a100_short --gres=gpu:1 --mem=16G --time=00:15:00 --pty bash
source cluster/env.sh && python cluster/check_setup.py
exit
```

Fix every `FAIL` line before going on.

## Timing run (one session)

```bash
mkdir -p logs
sbatch --array=1 cluster/run_sessions.sbatch
tail -f logs/spikes2lfp_<jobid>_1.out
```

The log has one line per trained model (test R2, best epoch, seconds) and per attributed model
(number of windows, seconds, completeness error), and the duration of each stage. Afterwards,
`sacct -j <jobid> -o JobID,Elapsed,MaxRSS,State` gives the wall time and peak memory: set `--time`
and `--mem` in `run_sessions.sbatch` from these, with some margin. If attribution is too slow, reduce
`ATTR_SAMPLED`, `ATTR_EVENTS` or `PERI_EVENTS` in `config.py` (the number of windows per model is
3 x ATTR_SAMPLED + 4 x ATTR_EVENTS + PERI_EVENTS x 63).

## All sessions

```bash
sbatch --array=1-89%8 cluster/run_sessions.sbatch   # at most 8 tasks at once
squeue -u $USER
```

A task that fails or runs out of time can be resubmitted with the same index
(`sbatch --array=5,17 cluster/run_sessions.sbatch`): finished models and attribution files are kept
and only the missing ones are computed. Don't run two tasks for the same session at the same time.

Other runs:

```bash
STAGES=train,transfer sbatch --array=1-89%8 cluster/run_sessions.sbatch      # models only
EXTRA_ARGS="--seeds 3" STAGES=train sbatch --array=1-10 cluster/run_sessions.sbatch   # seed variability
```

Outputs: `$SPIKES2LFP_DATA_ROOT/spikes2lfp_v2/{models,variables,attrs,plots}/<session_id>/`
(the first, published model version stays in `spikes2lfp/`).
