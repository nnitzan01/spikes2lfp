# Revision status: aims, reviewer comments, plan and progress

Working document for the revision of *Local and brain-wide spike correlates of lamina-specific local
field potentials in the visual cortex* (Nitzan, Ji & Buzsáki; Nature Communications, first round of
review). It is updated as each analysis is built, run and reviewed; the change log at the end
records what changed when.

Last updated: 2026-11-03.

---

## 1. The project

### Aim
Predict the laminar LFP of mouse VISp (broadband and 8 band-power envelopes, on 16 channels aligned to
the L4 border) from brain-wide single-unit spiking, then use attribution (integrated gradients) to
ask which neurons, areas, layers, cell types and functional clusters the predictions depend on, and
when (peri-stimulus time course, CSD components c1–c4).

### Data
- Allen Visual Behavior Neuropixels. 89 sessions with CSD-derived L4 borders
  (`tables/layer_4_borders.csv`, `cluster/sessions.txt`), from **48 mice** (41 recorded twice, 7
  once; `tables/sessions.csv`). The manuscript says 58 (Results) and 53 (Methods); both are wrong.
- Cre lines: Sst 34 sessions, Vip 21, wt 34. Optotagged units: 637 SST units in 41 sessions,
  55 VIP units in 20 sessions (the manuscript says 5 VIP).
- Unit table with curated metrics: `tables/units_info.csv` (stLFP, stPR, coupling index, coherence,
  speed / pupil scores, burst indices, cluster_id, layer, ...).

### Pipeline v1 (the submitted manuscript) and what was wrong with it
| Problem found in v1 | Consequence |
|---|---|
| Models had **no conv layer** (`conv=False`), unlike the Methods; the conv variant used centred padding (saw 1 bin into the future) | Methods mismatch |
| Attribution was **instantaneous**: input at t × ∂output(t)/∂input(t), hidden state held fixed | Nothing credited through the recurrence; completeness could not hold (R2 #2 is right) |
| Zero-baseline IG is nonzero only where a unit spiked; the L1 share was among the ~15 co-active units | Compositional measure; scores ~0.06 rather than ~0.001 |
| Spike scaling per time bin, not per neuron; LFP z-scoring and filtering fit on the whole epoch | Methods mismatch; possible leakage |
| Train/test = randomly shuffled 3 s chunks | Autocorrelation leaks across the split (R2 #5–6, R3 #2) |
| High-band envelopes downsampled to 250 Hz without anti-aliasing | 100–400 Hz content folded into 0–125 Hz |
| Linear baseline saw only the current bin and was fit on the last batch only | Explains why it missed the evoked bumps (R1) |
| Firing-rate control was bound to pass (per-spike measure; most removed spikes fell outside the window) | R3 #8 |
| Jaccard counted silent (tied) units; K = 40 in code vs 50 in text | Fig. 4D |
| Fig. 7B: r = −0.53 over 5 areas gives p = 0.36, text says p < 0.05 | R3 minor 8 |

### Pipeline v2 (branch `revision/history-aware-models`)
- **Model:** causal CNN-LSTM (CausalConv1d, 16 filters, kernel 3, left padding → LSTM 50 units →
  linear readout); a test checks that future inputs never change past outputs. 4 ms bins, 3 s
  sequences (750 bins). 9 targets per channel (broadband + 8 band envelopes); every VISp channel.
- **Split:** blocked; test = last 20% of the active epoch, validation = the 10% before it, 6 s
  buffers. Spike min-max scaling per unit and target z-scoring fit on training data only.
  Anti-aliased decimation 1250 → 250 Hz. Early stopping on validation (patience 5, max 15 epochs).
- **Attribution:** `SequenceIG`, history-aware IG through conv and recurrence over 3 s of history,
  Gauss-Legendre steps, completeness check (sum A = f(x) − f(b)), adaptive steps 50 → 200 → 400
  when the error exceeds 1%. Zero baseline (mean baseline supported, not run). Targets: 500 sampled
  bins each from train / test / spontaneous, CSD components c1–c4 of 500 presentations, peri-stimulus
  grid every 12 ms for 100 presentations. Also a "recent" total over the last 40 ms of lags.
- **Measures** (`analysis/sequence_attribution.unit_summaries`): mean, mean_abs, **per_spike**
  (main measure; closest to v1, ρ ≈ 0.5–0.8 with v1) and share (fraction of the model's total |A|,
  scaled by unit count). Lag kernels per area (`pooled/area_kernels.npz`).
- **Run:** all 89 sessions on NYU BigPurple; pooled with `experiments/pool_sessions.py` into
  `spikes2lfp_v2/pooled/`; figures from `pool_sessions.ipynb`.

---

## 2. Reviewer comments

### Reviewer 1
- Conclusion ("complex interplay between function, anatomy, firing statistics and coordination") is
  obvious; heavy on computation, thin on biology; not convinced it fits Nat. Commun.
- Better performance than linear methods, and better prediction of slow components, are expected.
- Visually driven vs spontaneous prediction: separate **evoked** (onset-locked) from **induced**
  components. Is the advantage over linear models just the evoked component? Why does the linear
  model miss the big bumps in Fig. 3B, and what lets the ML model capture them?
- Attribution in distant structures (midbrain) is the most interesting part, but would **simpler
  methods** have found the same (lower R², more interpretable)?
- SST vs VIP: state in the main text that cell type was determined **optogenetically**.
- Much of the paper is sanity checks (L6 predicts L6, LGN > CA1...).
- The 5 functional clusters × layers × frequencies × components is a 4D object with many
  comparisons whose biological meaning is unclear.
- Technical: no code-sharing statement; motivate the architecture in the main text; **sign**: a
  neuron predicting both positive and negative deflections could get ~0 attribution, like an
  uninformative one.

### Reviewer 2
- Conflates methodological and biological novelty. Functional groups organizing LFP relations
  beyond anatomy is not new (Nitzan & Buzsáki 2025; Jia et al. 2022; de Vries et al. 2020).
- **#1** Does attribution add insight beyond conventional measures (firing dynamics, spike–LFP
  coupling, population coupling)? Test directly.
- **#2** Is the procedure really IG on the full CNN-LSTM? With hidden state fixed per bin, earlier
  spikes get no credit through the recurrence. Specify gradient flow, hidden state along the path,
  which output is differentiated; compare with full-sequence IG; completeness / convergence check.
- **#3** Dividing by the L1 norm per bin and averaging over spike bins gives a fractional,
  competitive measure, not IG. Name it accordingly; report raw IG alongside and show conclusions
  don't depend on the normalization.
- **#4** The R² correction (regress total attribution on R² across 16 channels, redistribute
  residual) lacks justification and may remove signal. Present uncorrected attribution as primary;
  handle performance by sensitivity analyses (matched R², covariate); at minimum show effects exist
  before correction.
- **#5** Shuffled 3 s windows distribute nonstationarity across train/test instead of testing it;
  autocorrelation makes test data non-independent.
- **#6** "Split at the trial level" vs shuffled segments is unclear. Repeat with contiguous blocks,
  held-out trials, or independent sessions; show the attribution structure is preserved.

### Reviewer 3
- **#1 Stimulus clock.** High R² may come from the stereotyped evoked trajectory. Requests:
  performance after subtracting condition-specific PSTH / mean evoked LFP (predict single-trial
  residuals); compare with a stimulus-identity + time-since-onset predictor and a condition-average
  LFP baseline; shuffle spikes across trials keeping stimulus alignment; time-shift spikes; state
  how prediction is evaluated (concatenated, within trial, trial-averaged).
- **#2 Splits and leakage.** Overlap? Adjacent windows across sets? Repeated images in both sets?
  Normalization and filtering before the split leak. Requests: split schematic with counts;
  contiguous held-out blocks with buffers; preprocessing fit on training only; leave-session /
  leave-mouse-out where feasible.
- **#3 Baselines.** Linear model under-described; must get the same history. Requests: lagged ridge
  / Wiener filter, population rate, stimulus and behaviour predictors, CNN-only and LSTM-only;
  hyperparameter selection, early stopping, parameter counts, several seeds; exact R² definition
  with negative values kept.
- **#4 Transfer ≠ generalization.** Active → passive → spontaneous within a session is
  cross-condition transfer, not generalization. Report uncertainty and a null for R² ≈ 0.2. Use
  precise terminology; cross-session / leave-mouse-out if possible, otherwise state models are
  session-specific.
- **#5 Correlated inputs and confounds.** IG is model sensitivity, not unique information or
  biological source. Requests: incremental loss when each **region is removed** (inference-time and
  retrained); control for stimulus timing, population activity, running, pupil; **rank stability**
  across initializations and one other attribution / permutation method; predictive (not causal)
  language; don't read sign as de/hyperpolarization.
- **#6 High frequencies.** R² ≈ 0 above ~25 Hz, yet 50–400 Hz attribution is interpreted. Restrict
  inference to reliably predicted session × depth models; show performance with every attribution
  map; prespecified threshold; drop bands indistinguishable from a null.
- **#7 Normalization and correction.** L1 per bin + spike-conditional averaging is compositional
  and depends on unit counts. Requests: full equations and units; **validate on simulated data**
  with known sources; repeat with raw, time-averaged |A|, cumulative and per-spike attribution;
  fixed-size neuron subsamples and alternative IG baselines.
- **#8 Firing-rate control and ablation.** Selecting top-10 units then testing them is selection
  on extremes; Monte Carlo replicates aren't samples for a t-test. Retrained ablation lets
  correlated inputs compensate. Requests: select on one partition, evaluate on another;
  firing-rate-matched control neurons, inference over sessions / mice; inference-time and
  retrained ablation with repeated seeds and session-level uncertainty.
- **#9 Pseudoreplication / statistics.** Neurons and repeated sessions aren't independent.
  Requests: mouse/session summaries, hierarchical models or cluster bootstrap; CV grouped by
  session or mouse; effect sizes, CIs, exact p, n mice / sessions per analysis, multiple-comparison
  correction; define every ±.
- **#10 Clusters vs layers (0.404 vs 0.389).** Different problems, chance levels, samples; no
  paired test; ± undefined; clusters may exploit depth; cluster labels and attribution share
  information. Requests: same units and grouped folds; permutation null and chance-corrected
  metric; paired comparison; decode clusters within layers / matched laminar composition, cluster
  × layer table; independent trials for labels and attribution, k-means alignment across
  sessions; compare with firing rate, PSTH, depth, spike–LFP baselines.
- **#11 Spike bleed-through.** Report unit-to-contact distances, reference, filtering direction,
  artifact removal; repeat after excluding units near the contact and local VISp units; show
  high-frequency results aren't spike contamination.
- **#12 Fig. 8 random forest (R² 0.47).** Split structure undescribed; neuron-wise split leaks;
  predictors coupled to the target (stLFP, coherence, distance, clustering variables); impurity
  importance biased. Requests: leave-session / leave-mouse-out with CIs; permutation or conditional
  importance with stability; separate coupled from independent predictors.
- **#13 Cell types.** 533 SST vs 5 VIP is inadequate; label VIP descriptive; n mice / sessions per
  optotagged group; "putative" for waveform-defined pyramidal / PV.
- **Minor:** (1) define R² by equation, name it consistently; (2) Fig. 6E vs 6F reference;
  (3) neuron counts differ across Fig. 6 panels, state inclusion criteria; (4) CIs or
  session-level distributions for performance; (5) CSD details and reproducible L4 border
  procedure; (6) discuss whether L4-sink alignment favours consistent evoked motifs; (7) formal
  test of log-normality (QQ); (8) Fig. 7B hierarchy correlation: n areas, CI, influence, null;
  (9) LG/LGd vs LGN terminology, define abbreviations; (10) release code, configs, preprocessing,
  seeds, session IDs; (11) remove causal/mechanistic verbs throughout; (12) language edit.

---

## 3. Plan and status

Legend: ✅ done (run and reviewed) · 🟡 code ready / partly done · ⬜ not started · ❌ dropped

### Phase 1: rebuild the pipeline once
| Item | Answers | Status |
|---|---|---|
| Causal CNN-LSTM matching the Methods | R3 #3 | ✅ `models.py` |
| Blocked split with buffers, saved indices and counts | R2 #5–6, R3 #2 | ✅ `splits.py` (schematic figure still to draw) |
| Preprocessing fit on training only; anti-aliased decimation | R3 #2, #6, #11 | ✅ `preprocess_data.py` |
| Early stopping on validation; seeds supported; R² keeps negatives | R3 #3 | ✅ (`N_SEEDS = 1` so far) |
| History-aware IG with completeness check | R2 #2, R3 #7 | ✅ `integrated_gradients.py`, `analysis/sequence_attribution.py` |
| Attribution measures with equations (mean, mean_abs, per_spike, share) and recent-lag totals | R2 #3, R3 #7 | ✅ |
| Lag kernels per area | R1 (biology) | ✅ influence concentrated in the last ~50–100 ms, gone by ~300 ms |
| Within-session transfer to passive and spontaneous | R3 #4 | ✅ run; spontaneous R² ≈ 0 (see open issues) |
| Jaccard fix (silent units unranked) | R3 #5 | ✅ `analysis/ranking.py` |
| All 89 sessions run, pooled, figure notebook ported to v2 | — | ✅ |
| Session table (mouse, genotype) | R3 #9 | ✅ `tables/sessions.csv` |
| Behaviour / stimulus covariates on the spike-bin grid | R3 #1, #5 | 🟡 code ready (`covariates.py`, `covariates` stage); computed on first use by the experiments |

### Phase 2: the decisive analyses (set the framing)
| Item | Answers | Status |
|---|---|---|
| **A. Stimulus clock:** template per condition from training blocks, residual R², per-presentation R², shuffle-within-condition and time-shift nulls, model trained on residual, spikes + template | R1, R3 #1 | 🟡 one session run (`experiments/stimulus_clock.py`); 89 sessions to run |
| **B. Baselines:** lagged ridge (units instant / lagged, regions, stimulus, behaviour, combinations), region-rate LSTM (± behaviour) | R1, R3 #3 | 🟡 one session run (`experiments/baselines.py`); 89 sessions to run |
| B. CNN-only model | R3 #3 | ⬜ deferred |
| B. Repeated seeds, parameter counts, hyperparameter reporting | R3 #3 | ⬜ |
| **C. Does attribution beat simpler measures?** Rank units (per channel × target, training block) by mean \|attribution\|, \|attribution per spike\|, stLFP, spike–LFP correlation, population coupling, firing rate, lagged-ridge weight; remove the top 1–50% on the test block at inference time (mean or zero fill) vs random orderings; retrained ablation for broadband at 5% and 20%; Spearman of each measure with attribution | R2 #1, R1, R3 #8 | 🟡 code ready (`analysis/coupling_measures.py`, `experiments/ranked_ablation.py`), not run; session-level stats to add when pooling |
| Framing decision after A and C | R1, R2 | ⬜ |

### Phase 3: validation and robustness
| Item | Answers | Status |
|---|---|---|
| Ground-truth simulation (synthetic LFP from known real spike trains through laminar kernels with lags, shared arousal term, redundant units); score recovery for instantaneous IG, sequence IG, ridge, stLFP | R3 #7, R2 #2 | ⬜ `experiments/simulation.py` |
| Region removal: fixed list (VISp, other visual cortex, hippocampus, thalamus, midbrain, other + 16 single areas); inference-time removal and "region only" (mean / zero fill, targets ≤ 25 Hz); retrained without each global region (broadband), with and without covariates (stimulus onsets, running, pupil, licks, rewards, population activity). Do APN/MRN, DG/POST, subcortical results survive? | R3 #5 | 🟡 code ready (`experiments/region_ablation.py`), not run |
| Rank stability across seeds; permutation / occlusion importance; fixed-size unit subsamples; zero vs mean baseline | R3 #5, #7 | ⬜ |
| High-frequency: per session × channel × band null; units near the target contact and VISp excluded; unit-to-contact distances | R3 #6, #11 | ⬜ (`spectral_r2.py` has the unit-group part, not run) |
| Firing-rate control: select on training block, test on test block | R3 #8 | 🟡 `experiments/firing_rate_contribution.py` (not run); still compares to the area mean, not matched units |
| Ablation (fr / attribution / random ranking, retrained) | R3 #8 | 🟡 `experiments/ablation.py` (not run; one seed) |
| Spectral subsets (cortex, cortex without VISp, HPC, subcortical) | R3 #5 | 🟡 `experiments/spectral_r2.py` (not run) |
| R² correction as sensitivity analysis only | R2 #4, R3 #7 | 🟡 still in notebook cells 12–13; v2 R²–attribution correlation is positive (+0.2 to +0.65), unlike v1 (−0.8) |

### Phase 4: statistics and decoders
| Item | Answers | Status |
|---|---|---|
| Decoders with CV grouped by mouse, inner tuning grouped, within-session permutation nulls, firing-rate control, attribution + rate vs rate (`analysis/decoders.py`) | R3 #9, #10 | ✅ results below |
| Fig. 8 regression with grouped CV and held-out permutation importance | R3 #12 | ✅ results below |
| PYR vs INT decoding | R3 #13 | ❌ claim dropped (2026-11-03): firing rate alone beats attribution |
| Layer-controlled cluster decoder (within layers / matched composition, cluster × layer table) | R3 #10 | ⬜ tabled; supplementary at most |
| Fig. 8: separate predictors coupled to attribution (stLFP, coherence, distance) from independent ones | R3 #12 | ⬜ |
| `stats.py`: cluster bootstrap, mixed models (sessions within mice), FDR, CIs, n per panel | R3 #9 | ⬜ |
| Fig. 7B at the area level (n = 5, p = 0.36) or per-session slopes | R3 minor 8 | ⬜ |
| QQ plot for log-normality, or drop the claim | R3 minor 7 | ⬜ |
| VIP descriptive only, n mice / sessions per optotagged group | R3 #13 | ⬜ |

### Phase 5: text (after the analyses)
- One central claim instead of "complex interplay"; explicit novelty vs Nitzan & Buzsáki 2025,
  Jia 2022, de Vries 2020.
- Sign: signed vs absolute scores; what a mean-rate baseline changes.
- Predictive verbs; "putative" for waveform classes; SST/VIP optotagged; "within-session transfer".
- Methods: architecture rationale; full attribution equations and units; CSD details; L4-sink
  alignment caveat; split schematic; R² equation.
- Inconsistencies: mice 48 (not 58/53); R² 0.47 vs 0.43; 99.6% vs 99.8%; VIP 55 units not 5; K = 40
  vs 50; figure cross-references (4C/D, 5C → 5B, 6E → 6F); "binary" vs min-max spikes.
- Code release (GitHub + Zenodo), language edit.

---

## 4. Results so far (v2)

### Prediction (pooled, 89 sessions)
- Broadband test R² ≈ 0.7 on deep channels falling to ≈ 0.1 at the top, as in v1. Passive transfer
  matches v1.
- Slow bands lower than v1 (0.5–4 Hz deep ≈ 0.3 vs 0.5): causal model, single test block at the end.
- Spontaneous R² ≈ 0 (v1 ≈ 0.2); above zero only on channels 7–14.

### Attribution (per_spike)
- Replicates v1: area order VISp > LG > CA1; peri-stimulus maps by area, layer and cluster;
  cluster → CSD component mapping (cluster 1 at c1 … cluster 4 at c4), clearer than v1.
- Differs from v1: cross-band comparisons are dominated by 0.5–4 Hz (per_spike is in output units);
  thalamus and hippocampus have the opposite early sign in Fig. 7D; Jaccard dip later and broader
  (minimum ≈ 0.1 s); R²–attribution correlation positive.
- `share` largely restates firing rate (NOT, LG, APN, PPT on top; L5 > L6 > L4 > L2/3 everywhere;
  INT ≈ 2× PYR). Decision: per_spike is the main measure; share supplementary at most.

### Decoders (per_spike; 5 mouse-grouped folds, 48 mice; 50 within-session shuffles)
| | Chance | Attribution | Shuffle null | Firing rate alone | Attribution + rate |
|---|---|---|---|---|---|
| PYR vs INT | 0.50 | 0.687 | 0.499 | 0.935 | 0.905 (−0.030, 0/5 folds) |
| Layer (4) | 0.25 | 0.393 | 0.253 | 0.371 | 0.438 (+0.067 ± 0.008, 5/5 folds) |
| Cluster (5) | 0.20 | 0.412 | 0.208 | 0.251 | — |
| Fig. 8 R² | — | 0.302 ± 0.019 | −0.003 | — | — |
All p = 1/51 ≈ 0.02 (the minimum with 50 permutations).
- Mouse-grouped folds changed scores by < 0.01 vs unit folds: little mouse leakage.
- Cluster decoder's top features: 100–200 Hz attribution on the 4 deepest aligned channels, then
  0.5–4 Hz and broadband on the same channels (possible depth confound → layer-controlled decoder).
- Fig. 8: held-out permutation importance dominated by the VISp flag (0.33), then firing rate
  (0.06), distance from VISp, cluster_id. Under share: firing rate 1.26, VISp 1.10.

### Baselines (one session, 1047969464; median test R² over channels)
| Inputs | Broadband | 0.5–4 | 4–8 | 8–12 | > 12 Hz |
|---|---|---|---|---|---|
| Ridge, units, current bin only (v1-style linear) | 0.13 | 0.07 | 0.03 | 0.00 | ≈ 0 |
| Ridge, units, 1 s lagged | 0.54 | 0.20 | 0.12 | −0.01 | ≈ 0 |
| Ridge, 20 region counts, lagged | 0.43 | 0.19 | 0.10 | 0.06 | 0.00–0.04 |
| Ridge, stimulus onsets | 0.07 | 0.23 | 0.06 | 0.01 | 0 |
| Ridge, behaviour | 0.12 | 0.11 | 0.02 | 0.02 | 0 |
| Ridge, stimulus + behaviour | 0.21 | 0.27 | 0.07 | 0.07 | ≈ 0 |
| Ridge, regions + stimulus + behaviour | 0.48 | 0.29 | 0.12 | 0.11 | 0.00–0.05 |
| Ridge, units + stimulus + behaviour | 0.54 | 0.19 | 0.11 | 0.01 | ≈ 0 |
| CNN-LSTM, region counts | 0.34 | 0.12 | 0.07 | 0.02 | 0.00–0.03 |
| CNN-LSTM, region counts + behaviour | 0.36 | 0.16 | 0.06 | 0.01 | 0.00–0.04 |
- History is most of the linear model's performance (0.13 → 0.54): the v1 linear comparison was weak.
- Covariates add nothing to the lagged unit ridge (0.54 → 0.54); alone they reach 0.21 broadband
  and 0.27 at 0.5–4 Hz, more than the units for the delta envelope.
- Region counts get 80% of the unit ridge broadband R².
- The region-count CNN-LSTM is below the ridge on the same inputs (0.34 vs 0.43): see open issues.
| **CNN-LSTM, all units (the main model)** | **0.54** | **0.34** | **0.14** | **0.08** | 0.01–0.06 |
- The medians in this table are over all 25 channels, including 16–24, which lie above the
  registered end of the probe track (see below) and are never pooled. Over channels 1–15 broadband is **CNN-LSTM 0.587 vs lagged unit ridge 0.539**: higher on 13 of 15 channels,
  median paired gain +0.037; lower only on channels 14–15. The CNN-LSTM is clearly better for the
  band envelopes: 0.34 vs 0.20 at 0.5–4 Hz, 0.08 vs −0.01 at 8–12 Hz. One session; needs all 89.

### Stimulus clock (one session, 1047969464; 959 test presentations, 10 conditions)
| Median over channels | Broadband | 0.5–4 | 4–8 | 8–12 | 12–25 |
|---|---|---|---|---|---|
| Model R² | 0.538 | 0.336 | 0.141 | 0.077 | 0.022 |
| Template (stimulus clock) R² | 0.211 | 0.211 | 0.067 | 0.00 | 0.00 |
| Residual R² (model on what the template leaves) | 0.353 | 0.155 | 0.047 | 0.084 | 0.019 |
| Model retrained on the residual | 0.330 | 0.109 | 0.024 | 0.025 | 0.009 |
| Model given spikes + template | 0.554 | 0.335 | 0.151 | 0.090 | 0.032 |
| Spikes swapped between presentations of the same condition | −0.008 | 0.042 | 0.00 | −0.008 | −0.013 |
| Spikes shifted 30–120 s | −0.49 | −0.22 | −0.10 | −0.10 | −0.05 |
- Table medians are over all 25 channels. **Over the pooled channels 1–15** (broadband): model
  0.587, template 0.350, residual R² 0.361, retrained on the residual 0.336, spikes + template 0.593
  (+0.006), swapped spikes 0.095, shifted spikes ≈ −0.6.
- Broadband: the stimulus clock explains 0.35 of the model's 0.59 (60%); the model explains 36% of
  what is left, and a model trained only on the residual gets 0.34. Adding the template as an input
  gains < 0.02: the model already contains the stimulus clock. Swapping spikes between presentations
  of the same image drops R² to 0.10, far below the template: the prediction is trial-specific.
  Caveat: the swap also breaks slow state (arousal, running) along with the spikes; the baselines
  say behaviour adds nothing over the units, which limits this.
- 0.5–4 Hz envelope: mostly stimulus-locked (template 0.21 of the model's 0.34).
- Per-presentation R² is meaningful for broadband only (0.38 model vs 0.17 template); for the
  envelopes a 750 ms presentation is too short (negative for both).
- Channels 13–24 have no stimulus-locked response (template ≤ 0.07), yet channels 20, 22 and 23
  reach R² 0.54–0.66 while 19, 21 and 24 are ≤ 0.17. The channel table explains it: from channel 15
  up (probe 3400–3800 µm) every channel has the same CCF coordinates, i.e. lies beyond the
  registered end of the track, above the cortical surface. 16–24 were never pooled; 15 is now
  dropped too (decisions). 13–14 also lack an evoked response but are inside the track.

---

## 5. Open issues and decisions

### Decisions taken
- Channels past the registered probe track are not pooled. Checked in all 89 sessions: only
  1047969464 has any (channels 15–24, labelled VISp but with clamped CCF coordinates); its pooled
  range was cut from 1–15 to 1–14 in `tables/layer_4_borders_aligned.csv` (2026-11-03). Re-pool.
- Rebuild from scratch with causal models and history-aware IG; per-neuron min-max scaling fit on
  training; last 20% as test; all 9 targets attributed.
- per_spike is the main attribution measure.
- PYR vs INT decoding claim dropped (2026-11-03).
- Layer-controlled cluster decoder tabled (supplementary at most).

### Open
- **Spontaneous R² ≈ 0.** Hypothesis: targets scaled with the active-epoch scaler; add
  per-epoch z-scored R² to the transfer stage. Unconfirmed.
- Early sign flip of thalamus / hippocampus in Fig. 7D; possibly check with the mean baseline.
- Channel 13 of session 1047969464 was an outlier in the timing run.
- One model (session 1047969464, chan 4 band 2) above the completeness limit at 400 steps.
- Notebook run time: the cluster decoder with 50 permutations takes 8.3 h; the whole notebook ≈ 11 h
  of the 12 h `cpu_short` limit. Next rerun needs fewer cluster permutations or its own setting.
- High-band (> 25 Hz) results: keep with null + threshold, or cut. Not decided.
- `start_notebook.ipynb` still uses the old instantaneous IG.
- **Framing:** if the 89-session baselines confirm broadband CNN-LSTM ≈ lagged ridge, the paper can't
  argue that a nonlinear model is needed for broadband; the case rests on the band envelopes and on
  whether attribution tells us more than ridge weights (ranked ablation, ground-truth simulation).
- **Models may be undertrained.** Session 1047969464: 80% of broadband models (and about half at
  0.5–4 and 4–8 Hz) keep epoch 13 or 14 of 15, validation loss still falling (0.382 → 0.371 over the
  last five epochs). `experiments/training_length.py` retrains with a 40-epoch cap and also measures
  the cost of each 3 s sequence starting with an empty LSTM memory (R² after the first 1 s). If the
  gain is large, raise `NUM_EPOCHS` and retrain everything (models, attribution, experiments).
- Region-count CNN-LSTM below the ridge on the same 20 inputs (one session). Training on few,
  large-valued inputs may be failing; check before it is used as a baseline.

---

## 6. Running things (cluster cheat sheet)
- Repo: `/gpfs/home/nitzan01/Downloads/spikes2lfp`; data: `/gpfs/data/buzsakilab/nitzan01/Allen2022`;
  outputs: `.../spikes2lfp_v2/`; host `nitzan01@bigpurple.nyumc.org`. Always `source cluster/env.sh`.
- Sessions: `sbatch --array=1-89%8 cluster/run_sessions.sbatch`; check with
  `python -m experiments.pool_sessions check`; pool with `... pool`.
- Experiments: `EXPERIMENT=<name> sbatch --array=1-89%8 cluster/run_experiment.sbatch`
  (test one session first with `--array=1`).
- Figures: `pool_sessions.ipynb` via nbconvert on `cpu_short` (see `cluster/README.md`);
  `ATTR_MEASURE`, `DECODER_PERMUTATIONS` environment variables.
- Excluded bad GPU nodes: a100-4012, a100-4033, gn-0020.

---

## 7. Change log
- **2026-11-03** Session 1 best epochs: broadband models mostly stop at the 15-epoch cap. Added
  `experiments/training_length.py` (40-epoch retrain, warm-up-excluded R²).
- **2026-11-03** Channels past the track checked in all sessions; only session 1047969464 channel 15
  was pooled, now dropped. Pooling re-pools a session when its channel alignment changes.
- **2026-11-03** Channel table of session 1: channels 15–24 are above the registered track; one-
  session medians recomputed over the pooled channels 1–15.
- **2026-11-03** `stimulus_clock` ran on session 1 (1 h 22 min): the prediction is trial-specific
  beyond the stimulus clock (section 4).
- **2026-11-03** `baselines` ran on session 1 (1 h 13 min); results in section 4. Broadband CNN-LSTM
  equals the lagged unit ridge on this session; the envelopes are better with the CNN-LSTM.
- **2026-11-03** Experiment jobs failed: first a transient `module load` I/O error, then
  `GLIBCXX_3.4.30 not found` (scripts importing torch before scipy loaded the system libstdc++).
  Fixed in `cluster/env.sh` by preloading the environment's libstdc++.
- **2026-11-03** Region removal (Phase 3) built and tested on synthetic data.
- **2026-11-03** Ranked ablation (Phase 2C) built and tested on synthetic data; waiting for a
  one-session cluster test together with `stimulus_clock` and `baselines`.
- **2026-11-03** Document created. Grouped-CV decoders with nulls reviewed; PYR vs INT dropped;
  layer-controlled cluster decoder tabled.
