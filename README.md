# spikes2LFP

Predicting the multi-layer LFP of mouse primary visual cortex (VISp) from spiking activity across
visual and non-visual areas, and attributing the predictions to individual units.

Data: Allen Institute Visual Behavior Neuropixels (VBN), accessed through the AllenSDK S3 cache.
For each VISp channel and each target (broadband LFP and the power envelopes of 8 frequency bands),
a causal CNN-LSTM is trained on 4 ms binned spikes from the active task epoch. History-aware
integrated gradients then attribute each prediction to every unit at every lag of the preceding 3 s.

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
`<data_root>/spikes2lfp_v2/{models,variables,attrs,plots}/<session_id>/` (results of the first
model version are in `spikes2lfp/`):

```bash
SESSION=<data_root>/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>

python run_session.py $SESSION                       # all stages
python run_session.py $SESSION --stages attr,peri    # a subset (stages always run in pipeline order)
```

| Stage | Output |
|---|---|
| `train` | One causal CNN-LSTM per channel x target (x seed, `--seeds`); blocked split, training-only scaling, best-validation epoch; train/val/test R2 |
| `transfer` | R2 of the active-epoch models on passive replay and spontaneous activity |
| `plots` | Diagnostic figures of the trained models |
| `variables` | Scaled spike matrix, units, timestamps, stimulus table |
| `attr` | History-aware attribution (SequenceIG) of shared targets: random training/test/spontaneous bins, CSD-component windows, peri-stimulus grid; lag kernels; completeness errors |
| `summaries` | Per-unit mean, mean \|.\|, per-spike and share attribution for training, test, active and spontaneous targets |
| `peri` | PSTH, peri-stimulus ranks and top-K Jaccard stability (broadband) |
| `ipd` | Regional instantaneous predictive dominance |

Every stage skips outputs that already exist, so a failed cluster job can simply be resubmitted.
The module docstring of `run_session.py` lists every output file.

Experiments, run from the repo root after the relevant pipeline stages:

```bash
python -m experiments.detect_cortical_oscillations $SESSION   # HVS events (not used by the pipeline)
python -m experiments.ablation $SESSION --channel 5
python -m experiments.firing_rate_contribution $SESSION --channel 5
python -m experiments.spectral_r2 $SESSION
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `start_notebook.ipynb` | Step-by-step walk-through on one session: model comparison (Fig. 3B), training, attribution, Jaccard stability (Fig. 4D) |
| `pool_sessions.ipynb` | Pooled figures across sessions; starts with a section-to-figure map |
| `plots_for_manuscript.ipynb`, `simulate_lfp_for_figure.ipynb` | Example-session panels (Fig. 1A, 4A) and the Fig. 3D schematic |

## Notes on the method

- **Model:** Conv1d over time (16 filters, kernel 3) -> LSTM (50 units) -> linear readout at every bin.
  The conv is padded on the left only, so every prediction uses only current and past spikes
  (tested in `tests/test_models.py`).
- **Split:** the active epoch's 3 s chunks are split in time: training | validation | test (last 20%),
  with 6 s buffers dropped between sets (`splits.py`). Spike counts are min-max scaled per unit and
  targets z-scored, both with training statistics only. LFP targets are decimated to 250 Hz with an
  anti-alias filter.
- **Attribution:** for a target bin t, `SequenceIG` integrates gradients of the prediction at t with
  respect to the whole preceding 3 s window, through the conv and the recurrence. The attributions
  sum to f(x) - f(baseline) (completeness), and the error is saved for every target. With the zero
  baseline only spikes receive attribution; `IG_BASELINE = 'mean'` also credits silence.
- **Summaries** (`analysis/sequence_attribution.py`): per unit, the mean signed and mean absolute
  contribution per target, the contribution per spike, and the share of all units' |attribution|.
