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
| **A. Stimulus clock:** template per condition from training blocks, residual R², per-presentation R², shuffle-within-condition and time-shift nulls, model trained on residual, spikes + template | R1, R3 #1 | ✅ 89 sessions (section 4) |
| **B. Baselines:** lagged ridge (units instant / lagged, regions, stimulus, behaviour, combinations), region-rate LSTM (± behaviour) | R1, R3 #3 | ✅ 89 sessions (section 4); region-count LSTM only on session 1 |
| B. CNN-only model | R3 #3 | ⬜ deferred |
| B. Repeated seeds, parameter counts, hyperparameter reporting | R3 #3 | ⬜ |
| **C. Does attribution beat simpler measures?** Rank units (per channel × target, training block) by mean \|attribution\|, \|attribution per spike\|, stLFP, spike–LFP correlation, population coupling, firing rate, lagged-ridge weight; remove the top 1–50% on the test block at inference time (mean or zero fill) vs random orderings; retrained ablation for broadband at 5% and 20%; Spearman of each measure with attribution | R2 #1, R1, R3 #8 | 🟡 89 sessions run, with the mean-baseline attribution (`experiments/ranked_ablation.py`, `experiments/attribution_baseline.py`); ✅ 89 sessions with both IG baselines (section 4) |
| Framing decision after A and C | R1, R2 | ⬜ |
| **D. Local vs distributed sources by band (biology for R1):** from `region_ablation.npz`: R² drop by region × band (broadband, 0.5–4, 4–8, 8–12, 12–25 Hz) × depth from the L4 border, as a fraction of full R² and per unit, single visual areas, region alone; mean-baseline attribution share by region (`attribution_baseline.npz`) and spike–LFP correlation share for comparison. Control: rerun `region_ablation` with `--retrain-targets 0,1,2 --retrain-regions VISp,visual_cortex_other,hippocampus` (the band dissociation is inference-only so far) | R1 | ✅ 89 sessions, inference and retrained (section 4): the dissociation holds after retraining |
| **E. Evoked vs induced by region (biology for R1):** new `experiments/region_clock.py` on the stimulus-clock template: per region, R² and residual R² with the region mean-filled, and with only that region's spikes swapped between presentations of the same condition (keeps its evoked content, breaks its single-trial content); inference only | R1, R3 #1 | 🟡 built and tested 2026-11-10 (`experiments/region_clock.py`, `summarize region_clock`; also saves region shares of mean-baseline attribution, stLFP, spike–LFP correlation, ridge weight); ✅ 89 sessions (section 4) |
| **F. Region contributions to the task-evoked LFP (biology for R1; scoped 2026-11-11):** exact Shapley contributions of 6 region groups (VISp, other visual cortex, hippocampus, thalamus, midbrain, other; 64 coalitions, mean fill) plus single-area removal, over all active and passive bins; event-locked averages for repeated image, change (hit / miss, before the first lick), omission and the flash after it, familiar vs novel session and shared vs novel images, active vs passive replay; RMS share per region in early (30–100 ms), late (100–250 ms) and post (250–750 ms) windows; checked on the test block alone | R1, R3 #1 | 🟡 built and tested 2026-11-13: `experiments/region_events.py` (exact Shapley over the groups present, mean fill, active and passive; event-locked contributions all bins and test block; per-state sums for H; IG cross-check with the mean baseline on the same windows at L4, 100 events of image / change hit / miss / omission), `summarize region_events`; one-session test next |
| **G. Layer-specific inputs vs anatomy:** CSD (second spatial derivative across channels) of each source area's event-locked contribution, by depth from the L4 border, against the Allen connectivity atlas projection density of that area into VISp layers 1–6 (LG, LP, VISl, VISal, VISrl, VISam, VISpm; hippocampus and midbrain as no-projection controls); does each source match its own anatomical profile better than the others' (permutation over sources)? | R1 | ⬜ scoped |
| **Before F–H: data exploration (2026-11-11).** `experiments/explore_task.py` (figures in plots/<session>/explore/, `summarize explore_task` for the table across sessions and the familiar vs novel event-locked LFP): blocks and split, running, pupil quality and cleaning, licks, rewards and engagement, hit rate, event counts (all and test block), event-locked LFP by depth, band power, region firing rates and model predictions. 41 of 48 mice have one familiar (G) and one novel (H) session. **First two sessions (2026-11-13, 1047969464 G / 1048189115 H, same mouse):** mice run through the whole active epoch (stationary 0.2% of it, both sessions), so running vs stationary and pupil-within-stationary are not possible there; the pupil is clean in the active epoch (98–99% stable; outliers are fitting failures at block transitions) but has large frame-to-frame jitter (smooth ~1 s before use) and varies little within the active epoch (z ≈ −0.5, passive ≈ +1); engagement by 2 rewards/min: 89% vs 26% (hit rate 0.86 vs 0.53); test block holds only 6–19 misses and 45–47 omissions per session (hit / miss contrasts need pooling over sessions); the model reproduces the event-locked broadband and 0.5–4 Hz on the test block, including image vs change amplitude; firing is higher in passive (VISp baseline ≈ 6 vs 3 spikes/s); broadband LFP: L4 sink at ≈ 70 ms, deep positivity at ≈ 380 ms after the offset response (≈ 290 ms) and suppression; omissions flat; `is_image_novel` present (novel session: 668 familiar, ≥ 1000 novel flashes) **All 89 sessions (2026-11-13, `summarize explore_task`, table in variables/explore_task_sessions.csv):** stationary is a median 2.3% of the active epoch, > 10% in only 13 sessions (9 mice): running vs stationary is a subset analysis, running-speed deciles the main one; pupil stable in > 90% of the active epoch in 84 sessions; engaged a median 67% of the active epoch but only 38% of the test block (mice disengage towards the end; 63 sessions have both states, 20–80% engaged); hit rate 0.72 (2 sessions < 0.1: 1118324999, 1118512505); per session 162 hits, 65 misses, 177 omissions (test block 25 / 29 / 37); novel sessions have 880 familiar and ≥ 1000 novel flashes (test 200 / 630). Pooled layer 4 LFP (variables/explore_task_pooled.png): change > image in every band; novel sessions have a larger 70–250 ms broadband negativity for repeated images, active and passive; omissions carry no evoked response, only the 0.5–4 Hz decay. **Quantified (L4, mean 70–250 ms minus baseline, z):** repeated-image broadband −0.34 familiar vs −0.99 novel sessions; same mouse novel − familiar −0.63 (0/41 > 0, p 9e-13), passive −0.69 (0/41); within novel sessions, novel − shared images −0.47 (2/44 > 0, p 2e-12) and 200–400 Hz power +0.26 (43/44). Hit − miss: 0.5–4 Hz +0.47 (78/88 > 0, p 6e-14), broadband −0.09 (p 6e-07); running slows from ≈ 100 ms after changes (more on hits), so the hit − miss contrast may contain movement | R1 | ✅ 89 sessions |
| **H. Local vs distributed balance by brain state:** Shapley shares (Cov(c_r, prediction) / Var(prediction), which sum to 1) in running vs stationary (if the 89-session table shows any stationary time in the active epoch; otherwise running-speed quartiles), high vs low pupil (within stationary bins if possible, else within running bins) (pupil is noisy, with many NaNs and outliers: stable stretches only, short gaps interpolated; `explore_task.clean_pupil`), engaged vs disengaged (reward rate), active vs passive (per-epoch scaling); exploratory, two-sided | R1 | ⬜ scoped; depends on F |

**Already known from this dataset (don't present as new; 2026-11-13):**
- *Nitzan et al. 2024, Cell Reports 43:114521 (spikes, VBN):* novel images evoke stronger spiking in visual cortex,
  hippocampus and midbrain (less in thalamus), more sustained and with later onset; shared (familiar) images in
  novel sessions evoke weaker responses than on day 1, have lower hit rates (47% vs 74% novel), weaker change vs
  repeat enhancement (and its hierarchy gradient is lost), higher dimensionality, lower VISp → higher-area
  predictability of single-trial residuals, and less stable population vectors on misses; passive replay keeps
  these effects. So the larger L4 LFP negativity to novel images is the LFP counterpart of known spiking effects.
- *Piet et al. 2024, Neuron 112:1876 (2p, Visual Behavior):* mice mix a visual-comparison and a statistical-timing
  strategy (stable per mouse; dynamic logistic regression on lick-bout onsets), disengage over the session, and
  engagement has little effect on V1 population activity, while strategy does (Vip-Sst circuit, changes and
  omissions). Mice stop running to lick.
- **New for us:** the LFP (not spikes) view and, mainly, which regions' spiking the LFP novelty / change / hit
  signals come from (F), whether that follows the reduced VISp → higher-area communication for shared images,
  and whether region balance tracks strategy or state (H). Needed in region_events for this: shared images as
  their own event type in both sessions (same image, day 1 vs day 2), and a per-session strategy index.

### Phase 3: validation and robustness
| Item | Answers | Status |
|---|---|---|
| Ground-truth simulation: real spike trains, 40 driver units through causal alpha kernels (latency 0–200 ms), linear or rectified, ± a running-speed component no spike causes, AR(1) noise; CNN-LSTM trained and attributed (zero / mean baseline, recent lags); drivers recovered by IG vs ridge weight, stLFP, correlation, coupling, rate (AUC, precision, Spearman with true total and per-spike effect, credit to running-correlated non-drivers, kernel peak lag vs true latency) | R3 #7, R2 #2 | 🟡 run on session 1: ridge wins, but the CNN-LSTM underfits (R² 0.16–0.28); rerun with diagnostics, 40 epochs, 'gain' and 'burst' conditions |
| Region removal: fixed list (VISp, other visual cortex, hippocampus, thalamus, midbrain, other + 16 single areas); inference-time removal and "region only" (mean / zero fill, targets ≤ 25 Hz); retrained without each global region (broadband), with and without covariates (stimulus onsets, running, pupil, licks, rewards, population activity). Do APN/MRN, DG/POST, subcortical results survive? | R3 #5 | ✅ 89 sessions (section 4) |
| Rank stability across seeds; permutation / occlusion importance; fixed-size unit subsamples; zero vs mean baseline | R3 #5, #7 | ⬜ |
| High-frequency: per session × channel × band null; units near the target contact and VISp excluded; unit-to-contact distances | R3 #6, #11 | ✅ 89 sessions (section 4, 2026-11-13): units within 100 µm of each contact and all VISp units mean-filled and retrained away; high-band slow modulation survives |
| **High bands, timescale of prediction (2026-11-11):** R² of band power is dominated by brief transients the MSE-trained model can only predict on average, while it tracks the slower power modulation (manuscript Fig. 3 panel). `experiments/timescale_r2.py` (inference only): R² after smoothing target and prediction (4–250 ms), R² of log power and amplitude, coherence by modulation frequency, transient AUC and event-triggered averages, each against a 30–120 s time-shift null. Next: a ceiling from adjacent sub-bands (shared modulation, independent estimation noise); repeat without units near the contact (R3 #11) | R3 #6 | ✅ all 89 sessions (section 4): every band beats the time-shift null at every window; 100–400 Hz R² ≈ 0.5 at 250 ms. Ceiling and bleed-through repeat pending |
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

### Region removal (one session, 1047969464; broadband unless stated; median over pooled channels 1–14)
Full model 0.591. Drop in R² when the region is removed:
| Region | Units | At inference (mean fill) | Retrained | Retrained with covariates | Region alone (inference) |
|---|---|---|---|---|---|
| VISp | 21 | 0.111 | 0.056 | 0.052 | 0.094 |
| Other visual cortex | 304 | 0.231 | 0.115 | 0.090 | 0.327 |
| Hippocampus | 232 | 0.071 | 0.059 | 0.036 | 0.028 |
| Thalamus | 89 | 0.011 | −0.006 | 0.003 | 0.010 |
| Midbrain | 180 | 0.033 | 0.023 | 0.018 | 0.061 |
| Other | 23 | 0.011 | 0.001 | −0.006 | −0.012 |
- Covariates (stimulus onsets, running, pupil, licks, rewards, summed population) add nothing to the
  full retrained model (0.586 → 0.589) and shrink the drops only modestly: the regions carry
  information beyond stimulus timing, arousal and global activity.
- Per unit, VISp is the most important region (0.056 for 21 units vs 0.115 for 304 other visual
  cortex units), consistent with attribution ranking VISp first.
- Inference drops exceed retrained drops (redundancy: other units compensate when retrained).
  Zero fill gives much larger drops (other visual cortex 0.65): out of distribution; use mean fill.
- Hippocampus costs 0.036 even with covariates: possibly volume conduction into deep channels;
  look at it by depth when pooled. Among single areas VISl matters most (0.068 at inference).
- Run time 3 h 05 min per session (mostly the retraining).

### Ranked ablation (one session, 1047969464; median over pooled channels 1–14)
Drop in R² when the top units by each measure are removed on the test block (inference, mean fill):
| Broadband, fraction removed | 1% | 2% | 5% | 10% | 20% | Retrained 5% | Retrained 20% |
|---|---|---|---|---|---|---|---|
| Attribution (mean \|attr\|) | 0.055 | 0.083 | 0.170 | 0.301 | 0.481 | 0.041 | 0.138 |
| Attribution per spike | 0.058 | 0.089 | 0.173 | 0.251 | 0.358 | 0.073 | 0.097 |
| Ridge weight | **0.076** | **0.140** | **0.263** | **0.377** | **0.502** | **0.097** | **0.198** |
| Spike–LFP correlation | 0.015 | 0.064 | 0.216 | 0.362 | 0.494 | 0.051 | 0.195 |
| Population coupling | 0.013 | 0.028 | 0.081 | 0.177 | 0.308 | 0.019 | 0.098 |
| stLFP | 0.000 | 0.005 | 0.016 | 0.077 | 0.260 | 0.005 | 0.093 |
| Firing rate | 0.001 | 0.010 | 0.022 | 0.050 | 0.172 | 0.005 | 0.049 |
| Random | 0.002 | 0.005 | 0.011 | 0.025 | 0.068 | −0.003 | 0.021 |
- **Broadband: ridge weights find the units the CNN-LSTM depends on better than its own
  attribution**, at inference and retrained; spike–LFP correlation is as good as attribution from 5%.
  Consistent with broadband CNN-LSTM ≈ lagged ridge.
- **0.5–4 and 4–8 Hz envelopes (where the CNN-LSTM beats ridge): attribution is best or tied** at
  1–5% (0.5–4 Hz at 5%: attribution 0.191, correlation 0.191, ridge 0.138; 4–8 Hz: 0.197, 0.177, 0.177).
- Every measure is far above random; stLFP and firing rate are the weakest.
- Spearman with attribution (broadband): firing rate 0.81, correlation 0.59, ridge 0.58, per spike
  0.37, stLFP −0.14. Mean |attribution| is largely firing rate.
- Zero fill gives drops > R² (out of distribution) and orders the measures differently; not used.
- Possible mismatch: IG uses a zero (no spikes) baseline, so attribution credits each unit's mean
  drive, which the mean-fill ablation does not remove. IG with the training-mean baseline (also an
  "alternative IG baseline", R3) would match the test: `experiments/attribution_baseline.py`
  (one session, broadband to 4–8 Hz).

### Baselines over all 89 sessions (per-session median over pooled channels, then median over sessions)
| | Broadband | 0.5–4 | 4–8 |
|---|---|---|---|
| CNN-LSTM | 0.631 | 0.380 | 0.320 |
| Ridge, units + stimulus + behaviour | 0.600 | 0.339 | 0.271 |
| Ridge, units, 1 s lagged | 0.598 | 0.321 | 0.266 |
| Ridge, regions + stimulus + behaviour | 0.556 | 0.299 | 0.260 |
| Ridge, region counts | 0.499 | 0.170 | 0.136 |
| Ridge, stimulus + behaviour | 0.267 | 0.269 | 0.210 |
| Ridge, units, current bin only | 0.201 | 0.040 | 0.034 |
- CNN-LSTM minus lagged unit ridge (paired, per session): +0.027 / +0.028 / +0.042, higher in
  80 / 67 / 72 of 89 sessions, Wilcoxon p < 0.001 for each. Against the strongest ridge (units +
  stimulus + behaviour): +0.024 / +0.020 / +0.036, higher in 79 / 57 / 68 of 89 (p < 0.001).
  The nonlinear gain is small but consistent, and relatively largest in the slow bands; history is
  the larger factor (0.20 → 0.60 for the linear model).
- Stimulus and behaviour alone reach the model's level in no session for broadband (89 / 89 lower)
  and add ≈ 0.002 to the unit ridge.
- Unit identity matters: region counts lose 0.13–0.19 against the CNN-LSTM (87–88 of 89 sessions).
- Region-count CNN-LSTM: only session 1 (0.404 at 15 epochs); the other sessions ran with `--lstm none`.

### Training length (session 1047969464, channels 1–14)
- Raising the epoch cap from 15 to 40: broadband +0.014 (higher on 13 of 14 channels; best epoch
  median 24), 0.5–4 Hz −0.009, 4–8 Hz +0.007. Leaving out the first 1 s of each test sequence
  changes R² by ≤ 0.016. **Decision: keep the 15-epoch models**; report the 40-epoch gain as a
  sensitivity result. The CNN-LSTM's advantage over ridge is, if anything, underestimated.
- Region-count CNN-LSTM at 40 epochs: broadband 0.451 (best epoch 34, 43% in the last 3), 4–8 Hz
  0.304 (above the unit model's 0.247). It converges slowly; rerun it with `--lstm-epochs 60`.

### Attribution baseline (session 1047969464, channels 1–14)
Mean |attribution| with the training-mean IG baseline, in the ranked ablation (drop in R²):
| | 1% | 2% | 5% | 10% | 20% | Retrained 5% | Retrained 20% |
|---|---|---|---|---|---|---|---|
| Broadband, mean baseline | 0.073 | **0.144** | **0.268** | **0.394** | 0.496 | 0.065 | 0.167 |
| Broadband, zero baseline | 0.055 | 0.083 | 0.170 | 0.301 | 0.481 | 0.041 | 0.138 |
| Broadband, ridge weight | **0.076** | 0.140 | 0.263 | 0.377 | **0.502** | **0.097** | **0.199** |
| 0.5–4 Hz, mean baseline | **0.082** | **0.137** | **0.245** | **0.334** | **0.414** | | |
| 0.5–4 Hz, ridge weight | 0.043 | 0.056 | 0.138 | 0.268 | 0.340 | | |
| 4–8 Hz, mean baseline | **0.115** | **0.157** | **0.243** | **0.297** | **0.339** | | |
| 4–8 Hz, ridge weight | 0.071 | 0.113 | 0.177 | 0.238 | 0.285 | | |
- With the mean baseline, attribution ties with ridge weights for broadband and is clearly best for
  the 0.5–4 and 4–8 Hz envelopes at inference. Retrained, ridge weights stay ahead for broadband:
  attribution finds what this model uses, ridge what can't be replaced by other units.
- Mean-baseline per-spike attribution is poor (by construction: the baseline already contains the
  average drive). Completeness ≈ 0 at 50 steps; Spearman with the zero-baseline ranking 0.96 / 0.92 / 0.81.

### Stimulus clock over all 89 sessions (`summarize stimulus_clock`; median over sessions)
| | Broadband | 0.5–4 | 4–8 | 8–12 | 12–25 |
|---|---|---|---|---|---|
| Model | 0.631 | 0.380 | 0.320 | 0.133 | 0.157 |
| Template (stimulus clock) | 0.373 | 0.230 | 0.171 | 0.056 | 0.065 |
| Model − template (sessions > 0) | +0.248 (89/89) | +0.115 (83/89) | +0.122 (83/89) | +0.083 (79/89) | +0.091 (79/89) |
| Residual R² (share of the template's leftover the model explains) | 0.408 | 0.157 | 0.130 | 0.084 | 0.082 |
| Spikes swapped between same-image presentations | 0.153 | 0.101 | 0.036 | −0.044 | −0.019 |
| Spikes shifted 30–120 s | −0.635 | −0.368 | −0.298 | −0.171 | −0.170 |
| Retrained on the residual | 0.320 | 0.089 | 0.130 | | |
| Spikes + template as input, gain over model | +0.006 (60/89) | +0.032 (78/89) | +0.029 (80/89) | | |
- **The model is not a stimulus clock (R1, R3 #1).** The template accounts for 60–67% of the model's
  R² (broadband to 4–8 Hz); the model beats it in 89 / 83 / 83 of 89 sessions and explains 41% of
  the broadband variance the template leaves. With another same-image presentation's spikes it
  falls to 0.153, below the template itself: its predictions follow the single trial.
- Giving the model the template adds 0.006 broadband and ≈ 0.03 for 0.5–8 Hz: the model already
  captures nearly all of the stimulus-locked component.
- Per-presentation R² is negative for the band envelopes for both model and template (segments
  ≈ 0.75 s are short for slow envelopes); report it for broadband only (model 0.595, template 0.392).
- 8–12 and 12–25 Hz were retrained on session 1 only (`--retrain-targets 0,1,2`).

### Ranked ablation over all 89 sessions (`summarize ranked_ablation`; zero-baseline attribution)
Mean R² drop over 1–20% removed at inference; difference from ridge weight; sessions best of seven:
| | Broadband | 0.5–4 Hz | 4–8 Hz |
|---|---|---|---|
| Ridge weight | **0.331** (best 46) | **0.207** (best 34) | 0.165 (best 15) |
| Attribution | 0.324 (−0.019, 24/89; best 20) | 0.196 (−0.013, 33/89; best 24) | **0.196** (+0.019, 71/89; best 66) |
| Spike–LFP correlation | 0.321 (best 21) | 0.207 (−0.002, n.s.; best 30) | 0.155 |
| Attribution per spike | 0.287 | 0.147 | 0.130 |
| Population coupling | 0.168 | 0.075 | 0.064 |
| stLFP | 0.147 | 0.117 | 0.069 |
| Firing rate | 0.053 | 0.032 | 0.046 |
| Random | 0.017 at 10% | | |
- Attribution, ridge weight and spike–LFP correlation form the top group; stLFP, population
  coupling and firing rate are far behind (firing rate is near chance up to 5% removed).
- Spearman with attribution: firing rate 0.76 / 0.69 / 0.69, correlation 0.56 / 0.46 / 0.44, ridge
  0.54 / 0.42 / 0.35, stLFP ≈ 0.
- With the mean baseline (next section) attribution moves to the top in all three bands.

### Ground-truth simulation (session 1047969464, 3 draws of 40 drivers; `summarize simulation`)
| | Linear | Nonlinear | Linear + arousal | Nonlinear + arousal |
|---|---|---|---|---|
| CNN-LSTM test R² (ceiling ≈ 0.7) | 0.164 | 0.171 | 0.273 | 0.284 |
| AUC: ridge weight / IG zero / IG mean / correlation | 0.91 / 0.73 / 0.75 / 0.76 | 0.89 / 0.73 / 0.75 / 0.77 | 0.90 / 0.70 / 0.72 / 0.70 | 0.86 / 0.70 / 0.72 / 0.70 |
| Precision (top 40): ridge / IG mean | 0.68 / 0.36 | 0.68 / 0.31 | 0.65 / 0.28 | 0.64 / 0.28 |
| Spearman with true total: ridge / IG zero | 0.83 / 0.80 | 0.81 / 0.79 | 0.81 / 0.77 | 0.78 / 0.76 |
| Spearman with true per-spike effect: ridge / IG per spike | 0.82 / 0.61 | 0.82 / 0.58 | 0.82 / 0.54 | 0.82 / 0.55 |
| Confound percentile: ridge / IG zero / correlation (0.5 = none) | 0.64 / 0.76 / 0.80 | 0.67 / 0.77 / 0.81 | 0.75 / 0.85 / 0.98 | 0.77 / 0.87 / 0.98 |
| Peak lag vs true latency, IG zero: Spearman / median error | 0.20 / 98 ms | 0.08 / 106 ms | 0.13 / 110 ms | 0.04 / 109 ms |
- **As run, the simulation does not favour attribution:** ridge weights find the drivers far better
  (AUC ≈ 0.9 vs 0.7), recover per-spike effects better, and credit running-correlated non-drivers
  less. Lag recovery from the attribution kernels is near chance. Among the drivers, the
  zero-baseline attribution ranks true totals about as well as ridge (0.80 vs 0.83).
- **Not yet a fair test:** the CNN-LSTM reaches R² 0.16–0.28 where ≈ 0.7 is explainable, so its
  attributions come from an underfit model; and the generative model (a sum of linear lagged
  kernels, mildly rectified) is exactly the ridge's model class. Needed before using it: the ridge's
  R² on the same targets, the model's best epochs, a longer training cap, and a condition where
  the ridge is misspecified. See open issues.
- **Rerun, 'gain' and 'burst' (40-epoch cap, `simulation_nonlinear.npz`):**
  | | Gain | Burst |
  |---|---|---|
  | CNN-LSTM R² per draw (best epoch) | 0.452 (39), 0.218 (21), 0.034 (11) | 0.095 (20), 0.316 (37), 0.185 (10) |
  | Lagged unit ridge R² per draw | 0.152, 0.192, −0.317 | 0.228, 0.173, −0.069 |
  | AUC: IG mean / IG zero / ridge weight | 0.72 / 0.70 / 0.71 | 0.82 / 0.82 / 0.88 |
  | Precision: IG mean / ridge | 0.33 / 0.40 | 0.42 / 0.61 |
  | Spearman with true total: IG zero / ridge | **0.84** / 0.64 | 0.72 / **0.90** |
  | Confound percentile (no arousal term): IG zero / ridge / firing rate | 0.77 / 0.46 / 0.87 | 0.75 / 0.60 / 0.87 |
  | Drivers' attribution peak lag, median (true) | 40 ms (131) | 24 ms (131) |
  - The CNN-LSTM now beats the ridge (gain 0.235 vs 0.009 on average), far more than in the real
    data (+0.03), yet attribution only ties ridge weight at finding the gain drivers and loses for
    burst. Model fits are unstable across draws: two runs still improving at the 40-epoch cap, two
    stopped early (epochs 10–11) at low R². Negative ridge R² in draw 2 suggests the drivers' rates
    drift between the training and test blocks.
  - Attribution peaks are systematically too early (24–40 ms vs 131 ms true), not at lag 0 and not
    random: the model's learned timing is compressed. The lag claims (R2 #2) cannot rest on this yet.
  - IG credits high-rate non-drivers even with no arousal component (0.75–0.77); ridge does not.
- **Rerun, original conditions at a 40-epoch cap (`simulation_v2.npz`):**
  | | Linear | Nonlinear | Linear + arousal | Nonlinear + arousal |
  |---|---|---|---|---|
  | CNN-LSTM R² (15-epoch run) | 0.281 (0.164) | 0.245 (0.171) | 0.342 (0.273) | 0.314 (0.284) |
  | Lagged unit ridge R² | 0.119 | 0.052 | 0.203 | 0.187 |
  | AUC: IG mean / ridge | 0.87 / 0.91 | 0.83 / 0.89 | 0.83 / 0.90 | 0.79 / 0.86 |
  | Spearman with true total: IG zero / ridge | **0.92** / 0.83 | **0.89** / 0.82 | **0.90** / 0.81 | **0.86** / 0.79 |
  | Spearman with true per-spike effect: IG per spike / ridge | 0.80 / 0.80 | 0.78 / 0.81 | 0.80 / 0.80 | 0.73 / 0.80 |
  | Peak-lag Spearman, IG zero (median peak; true 131 ms) | 0.52 (68 ms) | 0.30 (24 ms) | 0.26 (50 ms) | 0.15 (16 ms) |
  - **Better-trained models give better attribution:** every attribution metric improved with the
    longer cap (AUC 0.73 → 0.87, lag Spearman 0.20 → 0.52 for 'linear'); among drivers, attribution
    now ranks true totals better than ridge and per-spike effects as well as ridge. Most best epochs
    are still at the cap (39), so the models have not converged.
  - The CNN-LSTM beats the ridge in every condition, by 0.08–0.23, even when the signal is linear:
    the ridge's 8 raised-cosine functions over 1 s cannot draw sharp kernels at 100–200 ms delays.
    Ridge weights still find the drivers best (AUC ≈ 0.9): detection does not need a good fit.
  - Both models are far below the nominal ceiling (≈ 0.7). Draws with negative ridge R² point to
    driver rates that drift into the test block; the noise-free ceiling on the test block is now saved.
  - Next: train to convergence (100-epoch cap, patience 10), and an LFP-like variant (200 drivers,
    kernel time constants 10–100 ms) to bring the R² levels and the CNN–ridge gap towards the real data.
- **Rerun to convergence (100-epoch cap, patience 10; `simulation_long_*`, `simulation_lfp_*`):**
  | | Lin | Nonlin | Lin + ar. | Nonlin + ar. | Gain | Burst |
  |---|---|---|---|---|---|---|
  | **Fast (40 drivers, τ 4–40 ms)**: CNN / ridge R² (ceiling ≈ 0.64–0.72) | 0.367 / 0.119 | 0.397 / 0.052 | 0.419 / 0.203 | 0.350 / 0.187 | 0.261 / 0.009 | 0.259 / 0.111 |
  | AUC: IG mean / ridge | 0.89 / 0.91 | **0.90** / 0.89 | 0.88 / 0.90 | 0.81 / 0.86 | **0.74** / 0.71 | 0.87 / 0.88 |
  | Spearman with true total: IG mean / ridge | **0.93** / 0.83 | **0.97** / 0.82 | **0.93** / 0.81 | 0.81 / 0.79 | **0.76** / 0.64 | 0.78 / **0.90** |
  | Peak-lag Spearman, IG mean (median peak; true 131 ms) | 0.65 (96) | 0.62 (130) | 0.65 (84) | 0.24 (16) | 0.13 (40) | 0.32 (64) |
  | **LFP-like (200 drivers, τ 10–100 ms)**: CNN / ridge R² (ceiling ≈ 0.72–0.75) | 0.533 / 0.417 | | | 0.392 / 0.441 | 0.343 / 0.341 | 0.425 / 0.096 |
  | AUC: IG mean / ridge | **0.83** / 0.78 | | | **0.77** / 0.74 | **0.72** / 0.64 | **0.80** / 0.77 |
  | Spearman with true total: IG mean / ridge | **0.91** / 0.76 | | | **0.79** / 0.74 | **0.69** / 0.50 | 0.70 / 0.70 |
  | Peak-lag Spearman, IG mean (median peak; true 136 ms) | 0.53 (136) | | | 0.18 (60) | 0.09 (44) | 0.09 (56) |
  - **Converged models give attribution at least as good as ridge weights.** Fast kernels: IG
    mean ties ridge at finding drivers (AUC within ±0.05; ridge ahead in the arousal and burst
    conditions) and ranks the drivers' true totals better in 5 of 6 conditions (burst: ridge).
    LFP-like: IG mean beats ridge weights in AUC in all four conditions (+0.03 to +0.08) and in
    ranking true totals in three (burst tied). IG per spike ranks per-spike effects better than
    ridge in 7 of 10 runs and worse in the two burst runs and nonlinear + arousal (fast)
    (fast 0.66–0.91 vs ridge 0.73–0.89; LFP-like 0.54–0.80 vs 0.57–0.69).
  - **The LFP-like regime brackets the real data:** CNN − ridge R² is +0.12 (linear), 0.00 (gain),
    −0.05 (nonlinear + arousal) and +0.33 (burst); real broadband +0.03. The fast regime always
    favours the CNN by 0.2–0.35 (ridge basis can't draw sharp kernels), so it is unrealistic.
  - **Lags only where the model fits well:** in linear and rectified worlds the drivers' attribution
    peaks match the true latency (median 130–136 ms vs 131–136, Spearman 0.5–0.65, error ≈ 25 ms);
    with gain, bursts or arousal + rectification they are compressed early (16–64 ms, Spearman
    ≤ 0.32). The R2 #2 lag claims need this caveat or only relative (early vs late) wording.
  - **Correlational measures fail:** STLFP AUC 0.38–0.70, spike–LFP correlation 0.57–0.77 and
    confounded by arousal (percentile 0.97–0.98); firing rate alone reaches AUC 0.80 in LFP-like
    burst (the condition rewards high rates), equal to IG mean there.
  - Training still varies across draws (best epochs 11–99 in one condition); single session, 3 draws.
- **Final run (LFP-like: 200 drivers, τ 10–100 ms; 10 draws × 3 sessions per condition; 100-epoch cap,
  patience 10; `summarize simulation`).** Means over 30 draws; paired differences over draws (median, draws > 0):
  | | Linear | Nonlin + arousal | Gain | Burst |
  |---|---|---|---|---|
  | CNN / ridge R² (ceiling ≈ 0.69) | 0.509 / 0.400 | 0.492 / 0.311 | 0.351 / 0.153 | 0.414 / 0.195 |
  | CNN − ridge, paired | +0.055 (23/30) | +0.086 (27/30) | +0.124 (28/30) | +0.081 (25/30) |
  | AUC: IG mean − ridge | **+0.023** (23/30) | **+0.031** (23/30) | **+0.067** (28/30) | **+0.057** (28/30) |
  | Spearman with true total: IG mean − ridge | **+0.129** (28/30) | **+0.131** (26/30) | **+0.218** (28/30) | +0.016 (17/30, n.s.) |
  | Spearman with true per-spike effect: IG mean − ridge | −0.019 (n.s.) | −0.026 (n.s.) | −0.035 (6/30) | −0.049 (8/30) |
  | Confound percentile: IG mean / ridge / spike–LFP correlation | 0.71 / 0.58 / 0.80 | 0.85 / 0.70 / 0.91 | 0.77 / 0.61 / 0.83 | 0.71 / 0.61 / 0.85 |
  | AUC − spike–LFP correlation, IG mean | +0.21 (30/30) | +0.17 (30/30) | +0.13 (30/30) | +0.12 (30/30) |
  | Peak-lag Spearman, IG mean | 0.34 | 0.23 | 0.11 | 0.14 |
  - **For the paper (R2 #1, R1 "simpler methods"):** where the ground truth is known and the CNN–ridge
    gap is near the real one (+0.06 to +0.12 vs +0.03 real), mean-baseline attribution finds the
    drivers better than ridge weights in all four conditions and ranks their total contribution better
    in three (burst tied). It beats spike–LFP correlation, stLFP, population coupling and firing rate
    everywhere (correlation: AUC +0.12 to +0.21, total-contribution Spearman +0.41 to +0.52, 29–30/30 draws).
  - **Limits to state:** ridge weights recover per-spike effects slightly better; attribution credits
    non-drivers that are high-rate or correlated with arousal more than ridge does (confound percentile
    0.71–0.85 vs 0.58–0.70), though less than spike–LFP correlation (0.80–0.91): attribution is the
    model's reliance, not causal drive. The zero-baseline per-spike measure is poor at detection
    (AUC 0.57–0.72), which supports the 2026-11-09 decision to use mean-baseline totals for importance.
  - **Lags:** peak-lag Spearman 0.11–0.34 in this regime. The attribution timing claims (R2 #2) cannot
    rest on peak lags; keep at most coarse early/late statements, or drop them.
- Confound credit is real for both: with an arousal component, units correlated with running are
  credited (IG 0.85–0.91, ridge 0.75–0.77, correlation 0.98). Attribution measures predictive
  reliance, not causal drive; the text must say so.

### Local vs distributed by band (`summarize region_bands`, 89 sessions; inference, mean fill)
Fraction of the full R² lost when the region is removed (median over sessions; full R² 0.631 / 0.380 /
0.320 / 0.133 / 0.157 for broadband / 0.5–4 / 4–8 / 8–12 / 12–25 Hz):
| Region (median units) | Broadband | 0.5–4 Hz | 4–8 Hz | 8–12 Hz | 12–25 Hz |
|---|---|---|---|---|---|
| VISp (49) | **0.56** | 0.32 | 0.26 | 0.23 | 0.35 |
| Other visual cortex (230) | 0.16 | **0.47** | **0.61** | **0.64** | **0.68** |
| Hippocampus (198) | 0.04 | 0.00 | 0.01 | 0.01 | 0.01 |
| Thalamus (110) / midbrain (112) | 0.01 / 0.01 | 0.02 / 0.05 | 0.04 / 0.05 | 0.06 / 0.09 | 0.05 / 0.08 |
| VISl / VISpm / VISal (single areas) | 0.04 / 0.01 / 0.02 | 0.09 / 0.05 / 0.05 | 0.11 / 0.08 / 0.07 | 0.11 / 0.08 / 0.06 | 0.14 / 0.08 / 0.11 |
- **A double dissociation, in nearly every session:** VISp − other visual cortex is +0.38 for broadband
  (76/89) and −0.17 to −0.40 for every band envelope; the broadband-minus-band interaction is positive in
  85–89 of 89 sessions (p < 1e-15). The local population carries the broadband (evoked, fast) LFP; the
  envelopes from 0.5 to 25 Hz depend mostly on the other visual areas (VISl, VISpm, VISal lead), i.e.
  the wider visual network. Not specific to slow bands: it holds through 12–25 Hz.
- Subcortical regions are minor but not zero for the envelopes (thalamus, midbrain, APN, LG 0.02–0.09
  at 8–25 Hz); hippocampus matters only for broadband (0.04).
- Depth (each channel divided by its own full R²; 0 = deepest pooled channel, L4 at about 7–10): VISp's
  broadband share peaks at and just below L4 (0.57–0.61 on channels 4–10, 0.27–0.33 at the edges). The
  other visual areas' 8–25 Hz share peaks in the middle (0.70–0.79 on channels 5–8); their 0.5–4 Hz share
  is lowest at L4 (0.41) and highest at both edges (0.59–0.64). Hippocampus loses 0.03–0.05 of broadband on
  the deep channels and 0.12–0.18 on the most superficial ones: rises away from the hippocampus, so not
  simple volume conduction from below.
- "Region alone" is negative for the envelopes (mean-filling most units is out of distribution); use
  only for broadband (VISp alone 0.61, other visual 0.25).
- **Retrained without the region (`region_ablation_bands`, 2026-11-13), fraction of full R² lost:**

  | Region | Broadband | 0.5–4 Hz | 4–8 Hz |
  |---|---|---|---|
  | VISp | **0.130** (89/89) | 0.114 (87/89) | 0.092 (86/89) |
  | Other visual cortex | 0.044 (89/89) | **0.166** (89/89) | **0.191** (88/89) |
  | Hippocampus | 0.037 (87/89) | −0.010 (36/89) | −0.013 (33/89) |

  Same ordering as at inference, smaller by 3–4× (retrained models recover from correlated units
  elsewhere; the inference losses overstate unique dependence). With covariates the numbers barely move
  (VISp 0.119 / 0.127 / 0.087; other visual 0.040 / 0.160 / 0.184). Paired VISp − other visual
  difference, retrained: broadband +0.084 (78/89, p 5e-12), 0.5–4 Hz −0.051 (33/89 > 0, p 3e-03), 4–8 Hz
  −0.080 (19/89 > 0, p 2e-09); the interaction (broadband − band) is +0.112 (82/89, p 7e-14) and +0.173
  (88/89, p 4e-16). **The double dissociation holds after retraining.**
  Hippocampus helps broadband only; removing it slightly helps the slow bands (fewer inputs to overfit).

### High bands by timescale (`summarize timescale_r2`; all 89 sessions, 2026-11-11)
| Test R² (median over sessions) | 0.5–4 Hz | 25–50 Hz | 50–100 Hz | 100–200 Hz | 200–400 Hz |
|---|---|---|---|---|---|
| No smoothing (4 ms) | 0.380 | 0.065 | 0.066 | 0.107 | 0.076 |
| Smoothed 50 ms | 0.389 | 0.093 | 0.138 | 0.316 | 0.374 |
| Smoothed 250 ms | 0.416 | 0.146 | 0.219 | 0.497 | 0.516 |
| Transient AUC (null) | 0.90 (0.59) | 0.75 (0.52) | 0.76 (0.52) | 0.81 (0.52) | 0.77 (0.52) |
| Coherence at 0–1 Hz modulation (null) | 0.34 (0.00) | 0.06 (0.00) | 0.14 (0.00) | 0.36 (0.01) | 0.42 (0.00) |
| Coherence at 1–4 Hz modulation (null) | 0.28 (0.09) | 0.14 (0.06) | 0.18 (0.07) | 0.42 (0.17) | 0.46 (0.12) |
| Coherence at 4–10 Hz modulation (null) | 0.05 (0.01) | 0.05 (0.01) | 0.09 (0.03) | 0.17 (0.06) | 0.26 (0.06) |
- **The model predicts the slow modulation of high-band power, not its fast transients.** 100–400 Hz
  R² rises from ≈ 0.1 unsmoothed to ≈ 0.5 at 250 ms, above the 0.5–4 Hz envelope (0.42; which barely
  changes with smoothing, as its power is already slow). Coherence is 0.36–0.46 below 4 Hz
  modulation, still 0.17–0.26 at 4–10 Hz, and ≈ 0.01–0.04 above 10 Hz. Bins with transients (> 3 SD)
  are ranked above the rest (AUC 0.75–0.81; null ≈ 0.52). **All 89 sessions beat the time-shifted
  spikes for every band and window** (the shifted R² is negative, so the coherence and AUC nulls are
  the useful thresholds).
- The 14-session preview held: every number within ≈ 0.02.
- 100–400 Hz is better predicted than 25–100 Hz at slow timescales (25–50 Hz is the weakest band at
  every window): consistent with high-frequency power tracking local spiking (multi-unit activity),
  which makes the spike bleed-through control (R3 #11, `bleed_through`) necessary before
  interpreting it.
- Log-power R² is negative from 8 Hz up (amplitude R² ≈ 0.01–0.11): a model trained on power does
  not fit the quiet periods that dominate the log scale; not a useful metric without log-power training.
- Broadband R² 0.631 unsmoothed rises to 0.779 at 100 ms and falls to 0.739 at 250 ms (smoothing
  then removes predictable 4–10 Hz content; coherence 0.65 there).
- **Proposed reporting for R3 #6:** keep all bands; report the unsmoothed R² with the 250 ms R², the
  1–4 Hz coherence and the transient AUC, each against the time-shift null.

### Spike bleed-through (`summarize bleed_through`, 89 sessions, 2026-11-13; R3 #11)
Units excluded per LFP channel by distance from the contact (same probe): nearest unit median 13 µm;
median 4 / 12 / 31 units within 50 / 100 / 200 µm, 49 VISp units. Test R² after 250 ms smoothing (median over sessions):
| | 0.5–4 Hz | 25–50 Hz | 50–100 Hz | 100–200 Hz | 200–400 Hz |
|---|---|---|---|---|---|
| All units (retrained reference) | 0.415 | 0.144 | 0.222 | 0.489 | 0.518 |
| Retrained without units within 100 µm | 0.404 | 0.139 | 0.217 | 0.491 | 0.444 |
| Retrained without any VISp unit | 0.372 | 0.123 | 0.206 | 0.431 | 0.393 |
| Paired change, no VISp (sessions > 0) | −0.055 (3/89) | −0.020 (8/89) | −0.020 (9/89) | −0.054 (2/89) | −0.106 (3/89) |
| Mean-filled at inference, no VISp | 0.240 | 0.076 | 0.137 | 0.242 | 0.141 |
- **The high-band prediction is not spike waveform bleed-through.** Without a single VISp unit, 100–200 Hz
  keeps 88% of its slow-timescale R² and 200–400 Hz 76%; coherence at 1–4 Hz modulation 0.42 → 0.37 and
  0.46 → 0.36; transient AUC 0.81 → 0.79 and 0.77 → 0.73. Removing the units within 100 µm costs nothing
  at 100–200 Hz and 0.054 at 200–400 Hz (paired; 0.074 → 0.064 unsmoothed).
- 200–400 Hz is the band that depends most on local units (as expected if it contains multi-unit power),
  but most of its predictable modulation is shared with distant populations.
- Mean fill at inference overstates the dependence (200–400 Hz 0.14 vs 0.39 retrained): the full model
  uses the nearest units when it has them, other units substitute when it is retrained without them.
  Report the retrained numbers.
- The models see sorted units only; unsorted spikes near the contact also reach the LFP and cannot be
  removed, so the control is conservative in the right direction.

### Evoked vs trial-specific by region (`summarize region_clock`, 89 sessions, 2026-11-13; inference)
R² lost when the region is mean-filled vs when only its spikes are swapped between presentations of the same
condition (keeps its evoked content, breaks its trial-specific content). Full R² 0.631 / 0.380 / 0.320 /
0.133 / 0.157; residual (template removed) R² 0.408 / 0.157 / 0.130 / 0.084 / 0.082.
| R² lost (median) | Broadband removed / swapped | 0.5–4 Hz | 4–8 Hz | 8–12 Hz | 12–25 Hz |
|---|---|---|---|---|---|
| VISp | 0.353 / 0.212 | 0.123 / 0.068 | 0.081 / 0.044 | 0.028 / 0.019 | 0.050 / 0.029 |
| Other visual cortex | 0.108 / 0.054 | 0.161 / 0.069 | 0.179 / 0.083 | 0.079 / 0.048 | 0.092 / 0.061 |
| Hippocampus | 0.024 / **0.043** | 0.000 / 0.004 | 0.003 / 0.005 | 0.001 / 0.003 | 0.001 / 0.003 |
| Thalamus / midbrain | 0.004 / 0.004, 0.007 / 0.006 | 0.007 / 0.006, 0.018 / 0.019 | | | |

Trial-specific fraction (swapped / removed, sessions where removing costs > 0.005):
| | Broadband | 0.5–4 Hz | 4–8 Hz | 8–12 Hz | 12–25 Hz |
|---|---|---|---|---|---|
| VISp | 0.69 | 0.55 | 0.60 | 0.76 | 0.60 |
| Other visual cortex | 0.53 | 0.42 | 0.49 | 0.68 | 0.58 |
| Hippocampus | 1.90 (85) | 0.59 (35) | 0.49 (40) | 0.52 (32) | 0.75 (23) |
| Midbrain | 0.71 | 0.80 | 0.41 | 0.56 | 0.84 |
- **Most of what each cortical region contributes is trial-specific, not the average evoked response:**
  swapping a region's spikes between presentations of the same image costs 40–75% of what removing it
  does. VISp's broadband contribution is the most trial-specific of the cortical ones (0.69 vs 0.53).
- **Hippocampus contributes to cortical broadband only through trial-specific activity:** swapping costs
  more than removing (0.043 vs 0.024, fraction 1.9; CA1 1.7, DG 1.9). Its spikes carry no stimulus-locked
  information the model uses; correctly timed, they help; mistimed, they mislead. Candidate mechanism for the
  text: hippocampal events (e.g. sharp-wave ripples, theta) reaching or co-occurring with cortical LFP.
- Per-region removal costs are significant in nearly every session for visual areas; thalamic and midbrain
  costs are small (≤ 0.02) but consistent (57–77 of 84–85 sessions).
- The "residual R² lost" rows exceed the residual R² for VISp broadband (0.56 vs 0.41): without VISp the
  model predicts trial-to-trial deviations worse than the template does.
- Region shares by other measures (median, broadband / 0.5–4 / 4–8 Hz): mean-baseline attribution VISp
  0.25 / 0.15 / 0.14, other visual 0.32 / 0.38 / 0.37, hippocampus 0.19 / 0.17 / 0.17; stLFP and
  spike–LFP correlation give other visual cortex 0.41–0.53 and VISp 0.14–0.18. Attribution and ridge
  credit hippocampus ≈ 0.19 although removing it costs ≈ 0 for the bands: shares measure use, not need.

### Transfer to passive and spontaneous epochs (`summarize transfer_scaling`, 89 sessions, 2026-11-13)
| Median R² | Broadband | 0.5–4 Hz | 4–8 Hz | 8–12 Hz | 12–25 Hz |
|---|---|---|---|---|---|
| Active test, active scalers | 0.631 | 0.380 | 0.320 | 0.133 | 0.157 |
| Passive, active scalers | 0.595 | 0.221 | 0.217 | 0.051 | 0.072 |
| Passive, recalibrated (first half) | 0.605 | 0.264 | 0.237 | 0.062 | 0.085 |
| Spontaneous, active scalers | 0.084 | −2.983 | −0.547 | −0.185 | −0.074 |
| Spontaneous, per-epoch z-score (v1) | 0.089 | −0.123 | −0.109 | −0.047 | −0.051 |
| Spontaneous, recalibrated | 0.216 | 0.003 | −0.006 | −0.002 | −0.001 |
| Spontaneous, r² | 0.215 | 0.027 | 0.005 | 0.004 | 0.000 |
- **Passive transfers well:** broadband 0.60 (active 0.63); the band envelopes lose 0.1–0.16 and
  recalibration recovers only 0.02–0.04 of it. Input rates are 1.27× the active training rate.
- **Spontaneous: the near-zero R² was partly scaling, partly real.** Broadband rises from 0.08 to 0.22 with
  a linear recalibration (gain and offset). The band envelopes stay at 0 however they are scaled
  (r² ≤ 0.03). In spontaneous (grey screen) the band power is lower and much less variable than in the
  active epoch (0.5–4 Hz target mean −0.70, SD 0.23 in active-scaler units); the model, trained on
  stimulus-driven power changes, does not track these smaller spontaneous fluctuations. Input rates are
  1.11× active, so not an input shift.
- Within the active test block, per-epoch z-scoring lowers R² (0.606) and recalibration changes nothing:
  the active scalers are right there.
- **For the paper:** report passive with active scalers and spontaneous with recalibration / r², and say the
  envelopes do not transfer to spontaneous activity. Open question: are spontaneous envelopes predictable
  at all from spikes (a model trained within the spontaneous epoch)? Not run.

### Attribution baseline over all 89 sessions (`summarize attribution_baseline`)
Inference ablation, mean fill: mean R² drop over 1–20% of units removed (median over sessions),
paired difference from ridge weight, and sessions where the measure is best of the six:
| | Broadband | 0.5–4 Hz | 4–8 Hz |
|---|---|---|---|
| Attribution, mean baseline | **0.360** (+0.017, 70/89, p 2e-8; best 63) | **0.235** (+0.033, 73/89, p 3e-11; best 61) | **0.227** (+0.047, 87/89, p 3e-16; best 84) |
| Attribution, zero baseline | 0.324 (−0.019, 24/89) | 0.196 (−0.013, 33/89) | 0.196 (+0.019, 71/89) |
| Ridge weight | 0.331 (best 18) | 0.207 (best 14) | 0.165 (best 2) |
| Spike–LFP correlation | 0.321 (−0.011, 26/89) | 0.207 (−0.002, 43/89) | 0.155 (−0.013, 28/89) |
| Attribution per spike (zero / mean baseline) | 0.287 / 0.207 | 0.147 / 0.157 | 0.130 / 0.130 |
| Random | 0.016 at 10% | | |
- **At inference, mean-baseline attribution is the best ranking in all three bands** and beats ridge
  weights in most sessions. Zero-baseline attribution is slightly worse than ridge for broadband and
  0.5–4 Hz and better for 4–8 Hz. Per-spike measures rank poorly here, as expected: removal tests
  total contribution, and per spike discounts the rate.
- **Retrained (broadband), ridge weight is ahead:** drop at 5% / 20% removed 0.073 / 0.173, vs
  mean-baseline attribution 0.069 / 0.140 (−0.003, 31/89; −0.027, 6/89). Spike–LFP correlation is best
  at 20% (0.196; +0.017, 70/89): removing every correlated unit takes away the redundant copies a
  retrained model would otherwise use. Attribution measures what this model relies on; ridge weights
  and correlation find information other units cannot replace. The paper should claim the former.
- Completeness error ≈ 0. Spearman of the mean-baseline ranking with the zero-baseline one
  0.95 / 0.94 / 0.94, with firing rate 0.65 / 0.48 / 0.51.

### Region removal over all 89 sessions (`summarize region_ablation`; R² drop, median over sessions)
| Region (median units) | Inference, broadband | 0.5–4 Hz | 4–8 Hz | Region alone | Retrained | Retrained + covariates |
|---|---|---|---|---|---|---|
| VISp (49) | 0.353 (89/89) | 0.123 | 0.081 | 0.385 | 0.084 (89/89) | 0.076 (89/89) |
| Other visual cortex (230) | 0.108 (89/89) | 0.161 | 0.179 | 0.154 | 0.029 (89/89) | 0.025 (88/89) |
| Hippocampus (198) | 0.024 (89/89) | 0.000 | 0.003 | 0.001 | 0.024 (87/89) | 0.016 (83/89) |
| Thalamus (110; 84 sessions) | 0.004 | 0.007 | 0.011 | ≈ 0 | 0.002 (56/84) | 0.002 (51/84) |
| Midbrain (112; 85 sessions) | 0.007 | 0.018 | 0.016 | ≈ 0 | 0.004 (60/85) | 0.001 (54/85) |
| Other (18; 83 sessions) | 0.000 | 0.001 | 0.001 | ≈ 0 | 0.001 | 0.000 (n.s.) |
- Single areas at inference (broadband / 4–8 Hz): VISl 0.023 / 0.038, VISal 0.012 / 0.024, VISpm
  0.009 / 0.021; every subcortical area ≤ 0.006 (DG 0.006, APN 0.003 / 0.009, MRN, SUB, POST, SC ≈ 0).
- **R3 #5:** the subcortical contributions (APN/MRN, DG/POST, thalamus, midbrain) are statistically
  detectable in many sessions but ≤ 0.01 R², and they shrink further once stimulus timing, behaviour
  and population activity are given to the model. Claims about them must be reworded or dropped.
- **Hippocampus is the exception:** removing it costs 0.024 broadband in 87 of 89 retrained sessions,
  0.016 with the covariates, but ≈ 0 for the 0.5–8 Hz envelopes, and it predicts nothing alone. Its
  contribution is to the broadband signal only (faster components?); see the depth analysis (open).
- Covariates add 0.001 to the full model; visual cortex carries the prediction (VISp alone 0.385).

---

## 5. Open issues and decisions

### Decisions taken
- Keep the 15-epoch models (40 epochs: broadband +0.014 on one session); report as sensitivity (2026-11-04).
- Channels past the registered probe track are not pooled. Checked in all 89 sessions: only
  1047969464 has any (channels 15–24, labelled VISp but with clamped CCF coordinates); its pooled
  range was cut from 1–15 to 1–14 in `tables/layer_4_borders_aligned.csv` (2026-11-03). Re-pool.
- Rebuild from scratch with causal models and history-aware IG; per-neuron min-max scaling fit on
  training; last 20% as test; all 9 targets attributed.
- per_spike is the main attribution measure.
- PYR vs INT decoding claim dropped (2026-11-03).
- **IG baseline (2026-11-09):** claims about which units the model depends on use mean-baseline
  total attribution (`attribution_baseline.npz`, best inference-time ranking in all three bands over
  89 sessions), worded as reliance of the model, not irreplaceable information (ridge weight and
  spike–LFP correlation lead when the model is retrained). Zero-baseline per_spike stays the
  measure for the descriptive maps.
- Layer-controlled cluster decoder tabled (supplementary at most).

### Open
- **Figure 6 is not in the repo:** it was made in MATLAB outside this codebase. Reviewer 3 asks only
  for text-level fixes on it (minor 2: the 6E vs 6F reference; minor 3: state why neuron counts differ
  across panels). If it has to be regenerated, the user can provide the MATLAB code to convert to Python.
- **Spontaneous R² ≈ 0.** No structural bug: the transfer stage feeds 3 s chunks as in training. The
  likely cause is the scaling: v2 scales the spontaneous spikes and targets with the active-epoch
  training scalers and sklearn's R² counts any offset or gain mismatch as error, whereas v1 z-scored
  the targets within each epoch. `experiments/transfer_scaling.py` (2026-11-11) scores active test,
  passive and spontaneous four ways (active scalers, per-epoch scaling as v1, linear recalibration
  on the first half, r²) with target / prediction means and SDs and the input rate ratio. **Answered
  2026-11-13 (section 4):** broadband 0.08 → 0.22 with recalibration; the band envelopes stay at 0.
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
- **Ground-truth simulation** (section 4): trained to convergence, IG mean matches or beats ridge
  weights; the LFP-like regime (200 drivers, τ 10–100 ms) brackets the real CNN–ridge gap.
  Proposed final run: LFP-like, 4 conditions (linear, nonlinear_arousal, gain, burst), 10 draws,
  3 sessions (`simulation_final_<condition>`, submitted 2026-11-10); `summarize simulation` pools it.
- Region-count CNN-LSTM: slow convergence (above). Rerun with `--lstm-epochs 60 --lstm-targets 0,1,2`.

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
- Excluded bad GPU nodes: a100-4012, a100-4033, gn-0020, gl40s-8009.

---

## 7. Change log
- **2026-11-13** Added `experiments/region_events.py` and `summarize region_events` (Phase 2 F, with the state sums for H and an IG cross-check).
- **2026-11-13** `bleed_through` over 89 sessions (run 11-12, jobs 29449362/3): section 4; R3 #11 analysis done.
- **2026-11-13** explore_task reviewed on two sessions (Phase 2 table); pupil example windows now lie inside the active epoch.
- **2026-11-13** `region_bands` (with the retrained run), `region_clock` and `transfer_scaling` finished for all
  89 sessions; section 4 updated. `summarize region_bands` now also tests VISp − other visual cortex on the
  retrained models. `bleed_through` session 1: 200–400 Hz R² at 250 ms 0.49 → 0.42 retrained without units
  within 100 µm, 100–200 Hz unchanged; full run with the VISp retrain next.
- **2026-11-11** Added `experiments/explore_task.py` and `summarize explore_task` (exploratory figures before Phase 2 F–H).
- **2026-11-11** Scoped Phase 2 F–H (region contributions to task events, layer match with anatomy, brain state).
- **2026-11-11** `timescale_r2` finished for all 89 sessions; section 4 updated (results match the 14-session preview).
- **2026-11-11** Added `experiments/bleed_through.py` and `summarize bleed_through` (R3 #11); test session now has
  probe positions.
- **2026-11-11** Added `experiments/timescale_r2.py` and `summarize timescale_r2` (high bands: what timescale is
  predicted, against the time-shift null).
- **2026-11-11** Response-to-reviewers draft for the settled points: `docs/RESPONSE_TO_REVIEWERS.md`,
  mirrored in the shared doc https://claude.ai/artifact/4nsq3YNy5vVbGLZDrYWD2M (keep in step).
- **2026-11-11** Spontaneous R²: no pipeline bug found; added `experiments/transfer_scaling.py` and
  `summarize transfer_scaling`. Noted that Figure 6 was made in MATLAB outside the repo.
- **2026-11-11** Final simulation (LFP-like, 30 draws per condition) and `region_bands` reviewed (section 4).
  Depth profile in `region_bands` now divides by each channel's own full R². `region_clock` and the
  `region_ablation_bands` retrain had not been submitted.
- **2026-11-10** Biology for R1 (Phase 2 D, E): `summarize region_bands` (band × depth × region from
  `region_ablation`), `region_ablation --output`, new `experiments/region_clock.py` and `summarize
  region_clock`; `stimulus_clock.clock()` factored out for reuse.
- **2026-11-10** Simulation trained to convergence, fast and LFP-like regimes, reviewed (section 4). Final
  LFP-like run submitted (4 conditions x 10 draws x sessions 1-3); `summarize simulation` without
  `--session` now pools draws over sessions with paired comparisons between measures.
- **2026-11-10** Simulation v2 reviewed (section 4). Added the noise-free ceiling, `--patience`, `--tau-ms`,
  `summarize simulation --pattern`.
- **2026-11-09** Decided: mean-baseline total attribution for the importance claims. Simulation: diagnostics
  (ridge R², best epoch, attribution peak lags), `--epochs`, `--output`, 'gain' and 'burst' conditions,
  counterfactual truth.
- **2026-11-09** Stimulus clock, ranked ablation (89 sessions) and the simulation reviewed (section 4).
- **2026-11-09** Attribution baseline and region removal reviewed over 89 sessions (section 4).
- **2026-11-09** All 89-session experiment runs finished (baselines, stimulus_clock, ranked_ablation,
  attribution_baseline, region_ablation); sessions re-pooled (89 sessions, 65,995 units). Baselines
  reviewed (section 4). Added the 89-session summaries to `experiments/summarize.py`.
- **2026-11-04** Ground-truth simulation built and tested on synthetic data (`experiments/simulation.py`;
  `python -m experiments.summarize simulation`).
- **2026-11-04** 55 `baselines` tasks failed within seconds on node gl40s-8009 (GPU unusable);
  node excluded in both sbatch files; resubmitting. Attribution baseline: 1 h 23 min per session.
  HPC confirmed a GPU hardware error and drained the node; the exclusion stays as a safeguard.
- **2026-11-04** Training-length, attribution-baseline (one session) and 34-session baselines
  reviewed (section 4). Baselines: `--lstm-epochs`, `--lstm-targets`. `attribution_baseline` pooled.
  Added `experiments/summarize.py`.
- **2026-11-03** Added `experiments/attribution_baseline.py`: training-mean IG baseline in the
  ranked ablation.
- **2026-11-03** `ranked_ablation` ran on session 1: ridge weights beat attribution for broadband;
  attribution best or tied for the 0.5–8 Hz envelopes (section 4).
- **2026-11-03** `region_ablation` ran on session 1 (3 h): visual cortex, VISp and hippocampus
  carry information beyond the covariates; thalamus none (section 4).
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
