# spikes2LFP

Predicting the multi-layer LFP of mouse primary visual cortex (VISp) from spiking activity across
visual and non-visual areas, and attributing the predictions to individual units.

Data: Allen Institute Visual Behavior Neuropixels (VBN), accessed through the AllenSDK S3 cache.
For each VISp channel and each target (broadband LFP and the power envelopes of 8 frequency bands),
an LSTM is trained on 4 ms binned spikes from the active task epoch. Integrated gradients then give
each unit's contribution to the prediction in every time bin.

## Setup

```bash
conda env create -f environment.yaml
conda activate spikes2lfp
pytest            # unit and end-to-end tests on synthetic data; no Allen data needed
```

## Layout

| Path | Contents |
|---|---|
| `config.py` | All shared settings: bin size, bands, LSTM architecture, training, IG and peri-event parameters |
| `paths.py` | Input/output directory layout; `DATA_ROOT` for the notebooks (env var `SPIKES2LFP_DATA_ROOT`) |
| `run_session.py` | The per-session pipeline (below) |
| `process_session.py`, `process_probe.py`, `preprocess_data.py` | Loading units, spikes, stimulus epochs and VISp LFP; binning, filtering, train/test chunking |
| `models.py` | `LSTMnet` (used throughout), `GRUnet`, `SpikingTransformer`, linear baseline, training wrapper |
| `integrated_gradients.py` | Integrated gradients for the recurrent models (`approx_param.py`, `captum_utils.py` are vendored from captum) |
| `analysis/` | Attribution summaries, per-bin ranking and top-K Jaccard, peri-event snippets and regional dominance |
| `spectral_analysis.py`, `plots.py` | Filters, band-wise R2, wavelet power; plotting helpers |
| `experiments/` | Stand-alone analyses: unit ablation, firing-rate control, area-group spectral R2, HVS detection |
| `tables/` | Curated unit list (`units_info.csv`), HVS session flags, VISp depths and layer-4 borders |
| `*.ipynb` | Exploration and figures (see below) |
| `tests/` | pytest suite |

## Running the pipeline

Scripts take a session directory from the AllenSDK cache and write results to
`<data_root>/spikes2lfp/{models,variables,attrs,plots}/<session_id>/`:

```bash
SESSION=<data_root>/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>

python run_session.py $SESSION                       # all stages
python run_session.py $SESSION --stages attr,peri    # a subset (stages always run in pipeline order)
```

| Stage | Output |
|---|---|
| `train` | One LSTM per channel x target (loaded if already saved; `--retrain` to force); test/train R2, losses, train/test split |
| `plots` | Diagnostic figures of the trained models |
| `variables` | Scaled spike matrix, units, timestamps, stimulus table |
| `attr` | Raw attributions for each window in `config.ATTR_WINDOWS` (active 720 s, spontaneous 300 s, whole session) |
| `summaries` | Per-unit mean, relative and per-spike (IPI) attribution for the active and spontaneous windows |
| `peri` | Peri-stimulus and peri-HVS snippets: PSTH, ranks, top-K Jaccard, mean attribution and per-spike potency |
| `ipd` | Regional instantaneous predictive dominance |

Every stage skips outputs that already exist, so a failed cluster job can simply be resubmitted.
The module docstring of `run_session.py` lists every output file.

Experiments, run from the repo root after the relevant pipeline stages:

```bash
python -m experiments.detect_cortical_oscillations $SESSION   # before the 'peri' stage, for HVS snippets
python -m experiments.ablation $SESSION --channel 5
python -m experiments.firing_rate_contribution $SESSION --channel 5
python -m experiments.spectral_r2 $SESSION
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `start_notebook.ipynb` | Step-by-step walk-through on one session; compares linear, GRU, LSTM and transformer models |
| `analyze_attribution_single_session.ipynb` | Rank, Jaccard and dominance analyses of one session |
| `pool_sessions.ipynb` | Pooling results across sessions (reads per-session outputs gathered into shared folders) |
| `plots_for_manuscript.ipynb`, `simulate_lfp_for_figure.ipynb` | Manuscript figures |

## Notes on the method

- Spike input is min-max scaled per time bin across units (`pre_process_spikes.minmax`). With 4 ms
  bins this is close to a binary spike indicator. It does not remove firing-rate effects from
  attributions; the per-spike (IPI) summaries do.
- With a zero baseline, a unit has non-zero attribution only in bins where it spiked. Silent units
  are never ranked, so they cannot enter top-K sets.
- The LFP is downsampled from 1250 Hz to 250 Hz without an anti-alias filter (kept so existing
  models stay comparable).
