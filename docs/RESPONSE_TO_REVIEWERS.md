# Response to reviewers — draft

Draft of 2026-11-11. Working copy of the shared doc https://claude.ai/artifact/4nsq3YNy5vVbGLZDrYWD2M;
keep the two in step (edit one, copy to the other). Results behind every number are in
`docs/REVISION_STATUS.md`, section 4.

## Scope of this draft

This draft answers the comments whose analyses are finished on all 89 sessions; the rest are listed under Pending with what they wait on. Numbers are medians over sessions of the per-session median over the pooled VISp channels, with paired Wilcoxon tests over sessions, unless stated otherwise.

Text in the responses is written to the reviewers. Notes for the authors are marked **[Authors]** and must be removed before submission. Figure and section numbers of the revised manuscript are placeholders (**Fig. Sx**) until the figures are made.

## Summary of the main changes

We rebuilt the pipeline from scratch to address the methodological concerns, reran every analysis on all 89 sessions, and added the controls the reviewers asked for. The main changes:

1. **Causal model.** A causal CNN-LSTM (left-padded convolution, then LSTM); a test confirms that future inputs never change past outputs. The submitted model had no convolutional layer, contrary to the Methods.
2. **Blocked splits, training-only preprocessing.** The test set is the last 20% of the active epoch, validation the 10% before it, with 6 s buffers. Spike scaling (per unit) and target z-scoring are fit on training data only; band envelopes are anti-alias filtered before decimation. Early stopping on validation loss.
3. **History-aware integrated gradients.** Attribution is computed through the convolution and the recurrence over the full 3 s of input history, with a completeness check (attributions sum to f(x) − f(baseline)) and adaptive integration steps. The submitted attribution held the hidden state fixed and so gave no credit through the recurrence (Reviewer 2 #2).
4. **Fair baselines.** Lagged ridge regressions with 1 s of history on units, region counts, stimulus and behaviour, and their combinations (Reviewer 1, Reviewer 3 #3).
5. **Stimulus-clock controls.** A condition-specific template, single-trial residuals, within-condition spike swaps and time shifts (Reviewer 1, Reviewer 3 #1).
6. **Attribution validated.** Against simpler measures by ranked removal on all sessions, and against ground truth in a simulation built from the recorded spike trains (Reviewer 2 #1, Reviewer 3 #7).
7. **Region removal** at inference and after retraining, with and without stimulus, behaviour and population covariates (Reviewer 3 #5).

**[Authors]** Claims that did not survive: the advantage of the nonlinear model for broadband is small (+0.03 R²); subcortical contributions are ≤ 0.01 R²; the pyramidal-vs-interneuron decoding claim is dropped; attribution peak lags are not reliable enough to support timing claims.

## Reviewer 1

**1.1 The conclusion is obvious; heavy on computation, thin on biology.**

We agree, and have refocused the paper on two findings the new controls make possible.

- **Local and distributed sources split by frequency.** Removing primary visual cortex (VISp) units costs 56% of the broadband R² but only 23–35% for the band envelopes. Removing the other visual areas does the opposite: 16% of broadband but 47–68% for every envelope from 0.5 to 25 Hz. The difference between broadband and envelopes is positive in 85–89 of 89 sessions (p < 10⁻¹⁵). The local population carries the broadband signal; the envelopes depend on the wider visual network, led by VISl, VISpm and VISal (**Fig. Sx**).
- **The model follows single trials, not only the stimulus.** A stimulus-locked template explains 60–67% of the model's R², but the model beats it in 89, 83 and 83 of 89 sessions (broadband, 0.5–4, 4–8 Hz) and explains 41% of the broadband variance the template leaves (point 1.3).

**[Authors]** The frequency split is so far measured with the trained models (units replaced by their mean). The retrained check for 0.5–8 Hz is running; keep this point conditional until it returns. Which regions carry the trial-specific part (point 1.3) is also running.

**1.2 Better performance than linear methods, and on slow components, is expected.**

We now compare against a linear model with the same history: a ridge regression on every unit's spikes over the preceding 1 s. The submitted linear baseline saw only the current time bin, which made it a weak comparison. Against the fair baseline the CNN-LSTM's advantage is small but consistent: +0.027, +0.028 and +0.042 R² (broadband, 0.5–4, 4–8 Hz), higher in 80, 67 and 72 of 89 sessions (p < 0.001 each). History is the larger factor: the linear model goes from 0.20 to 0.60 broadband when given 1 s of history. We have rewritten the Results to make that the point, rather than the nonlinear model's superiority.

**1.3 Evoked vs induced components; why does the linear model miss the large deflections in Fig. 3B?**

We built a stimulus-clock predictor: the mean LFP for each image condition and time since onset, fitted on training data. It reaches R² 0.373 broadband against the model's 0.631. Three tests show the model's advantage is trial-specific:

1. On the residual after subtracting the template, the model explains 41% of the remaining broadband variance (16% and 13% for 0.5–4 and 4–8 Hz).
2. Swapping each presentation's spikes with another presentation of the same image drops the model to 0.153, below the template itself.
3. Giving the model the template as an extra input adds only 0.006 broadband: it already captures the stimulus-locked component.

The large deflections in Fig. 3B were missed because the submitted linear baseline saw only the current 4 ms bin (R² 0.20). With 1 s of history the linear model captures them (R² 0.60).

**1.4 Would simpler methods find the same units, particularly in distant structures?**

We tested this two ways.

- **On the recordings:** we ranked units by each measure and removed the top 1–20% from the trained model. Removing the units ranked by attribution (mean baseline) costs more R² than removing those ranked by ridge weight, spike–LFP correlation, spike-triggered LFP, population coupling or firing rate, in all three bands (best ranking in 63, 61 and 84 of 89 sessions).
- **On simulated data with known sources** (real spike trains driving a synthetic LFP): attribution finds the true drivers better than ridge weights (AUC +0.02 to +0.07 across four generative models) and far better than spike–LFP correlation (+0.12 to +0.21, in 30 of 30 draws).

For distant structures the answer changes our conclusions: removing thalamus or midbrain costs at most 0.01 R², and less once stimulus timing, running, pupil and population activity are given to the model. We have removed the claims about subcortical contributions and describe them as detectable but minor.

**1.5 State that SST and VIP identity was determined optogenetically.** Done in the main text.

**1.6 Much of the paper is sanity checks.** We moved the expected results (layer 6 predicts layer 6, LGN above CA1) to the supplement and shortened them.

**[Authors]** Confirm 1.5 and 1.6 when the text is revised.

**1.7 Functional clusters × layers × frequencies × components has many comparisons of unclear meaning.** See Pending.

**1.8 Technical points.**

- *Code:* the full pipeline will be released with the paper, with configurations, seeds and session IDs.
- *Architecture:* the model is now motivated and described in the main text (causal convolution for local temporal features, LSTM for longer history).
- *Sign:* a unit that predicts both positive and negative deflections no longer averages to zero. The importance claims use mean |attribution| (with a training-mean baseline), and the signed mean is reported separately.

## Reviewer 2

**2.0 Methodological and biological novelty are conflated; functional groups organising LFP relations beyond anatomy are not new.**

We have separated the two. The methodological contribution is a causal model with history-aware attribution, validated against ground truth and against simpler measures (2.1, 2.2). The biological contribution is narrower than before and now stated against Nitzan & Buzsáki (2025), Jia et al. (2022) and de Vries et al. (2020): the split between local broadband and network-wide band-envelope sources, and the trial-specific component beyond the stimulus clock (Reviewer 1, point 1.1).

**2.1 Does attribution add insight beyond conventional measures (firing dynamics, spike–LFP coupling, population coupling)?**

We tested this directly on the recordings and on simulated data with known sources.

- **Ranked removal (89 sessions).** Units were ranked by each measure on the training block and the top 1–20% removed from the trained model on the test block. Attribution with a training-mean baseline was the best ranking in all three bands: mean R² drop 0.360, 0.235 and 0.227 (broadband, 0.5–4, 4–8 Hz), above ridge weights by +0.017, +0.033 and +0.047 (70, 73 and 87 of 89 sessions; p < 10⁻⁷). Spike–LFP correlation was close to ridge weights; spike-triggered LFP, population coupling and firing rate were far behind.
- **Ground truth.** Real spike trains of 3 sessions drove a synthetic LFP through 200 driver units, under linear, rectified-with-arousal, multiplicative-gain and burst-dependent generative models (10 draws each). Attribution found the drivers better than ridge weights in all four (AUC +0.02 to +0.07) and ranked their total contribution better in three (Spearman +0.13 to +0.22; tied for burst). It beat spike–LFP correlation in 29–30 of 30 draws in every model.
- **Limits.** When the model is retrained without the removed units, ridge weights and spike–LFP correlation lead: they find information other units cannot replace, whereas attribution measures what the trained model relies on. In simulation, attribution gives more credit than ridge weights to high-rate or arousal-correlated non-drivers, though less than spike–LFP correlation does. We word attribution as predictive reliance throughout, not as causal drive.

**2.2 Is it really IG on the full CNN-LSTM?**

In the submitted version it was not: the hidden state was held fixed, so earlier spikes received no credit through the recurrence. We thank the reviewer for catching this. The revised attribution integrates the gradient of the output at a target bin with respect to every input bin in the preceding 3 s, through the convolution and the LSTM, along a straight path from the baseline (Gauss–Legendre quadrature). The hidden state is recomputed along the path from a zero initial state at the start of the 3 s window. The completeness error (sum of attributions vs f(x) − f(baseline)) is checked for every model and the number of steps raised from 50 to 200 or 400 until it is below 1%; the median error is about 0.

**[Authors]** Add the equations and the number of models needing more steps to the Methods.

**2.3 L1 normalisation per bin is a fractional, competitive measure, not IG.**

We agree and have removed it. The measures are now the raw attributions in output units: the mean |attribution| per unit and the attribution per spike. Their equations and units are in the Methods. The share measure (fraction of the total) is kept only in the supplement and named as such. The conclusions do not depend on the IG baseline: rankings with a zero and a training-mean baseline correlate at Spearman 0.94–0.95.

**2.4 The R² correction lacks justification and may remove signal.**

Uncorrected attribution is now the primary result. **[Authors]** The correction is still in the figure notebook; decide whether to drop it or show it as a sensitivity analysis. Note that in v2 the attribution–R² relation across channels is positive (+0.2 to +0.65), unlike v1 (−0.8).

**2.5 Shuffled 3 s windows distribute nonstationarity across train and test.**

The split is now blocked in time: the last 20% of the active epoch is the test set and the 10% before it the validation set, with 6 s buffers between them. All preprocessing is fit on the training block. Stimulus-clock nulls (spike swaps within condition, 30–120 s time shifts) confirm the test performance depends on the correct single-trial pairing of spikes and LFP.

**2.6 Repeat with contiguous blocks, held-out trials or independent sessions.**

Every result in the revision uses the contiguous held-out block. The attribution structure replicates the submitted version (area order VISp > LGN > CA1, cluster-to-CSD-component mapping). **[Authors]** Leave-session-out is not feasible with session-specific inputs (different units per session); say so.

## Reviewer 3

**3.1 High R² may come from the stereotyped evoked trajectory.**

We ran each requested control on all 89 sessions. Prediction is evaluated on the concatenated held-out block; per-presentation R² is reported for broadband.

| Test-block R² (median over 89 sessions) | Broadband | 0.5–4 Hz | 4–8 Hz |
| --- | --- | --- | --- |
| CNN-LSTM | 0.631 | 0.380 | 0.320 |
| Template: mean LFP per image condition and time since onset | 0.373 | 0.230 | 0.171 |
| Stimulus identity, onsets and behaviour (lagged ridge) | 0.267 | 0.269 | 0.210 |
| Model on the residual (data minus template) | 0.408 | 0.157 | 0.130 |
| Model retrained to predict the residual | 0.320 | 0.089 | 0.130 |
| Spikes swapped between presentations of the same image | 0.153 | 0.101 | 0.036 |
| Spikes shifted by 30–120 s | −0.635 | −0.368 | −0.298 |

The model beats the template in 89, 83 and 83 of 89 sessions. With spikes from another presentation of the same image it falls below the template, so its prediction depends on the single trial. Per-presentation broadband R² is 0.595 for the model and 0.392 for the template.

**3.2 Splits and leakage.**

The split is now blocked in time with 6 s buffers (Reviewer 2, point 2.5); scaling, z-scoring and filtering are fit on the training block; decimation is anti-aliased. Images repeat across the blocks by design of the task, which is why we added the within-condition spike swap above. A split schematic with counts is in **Fig. Sx**. Decoders and the Fig. 8 regression now use folds grouped by mouse (48 mice); grouping changed scores by less than 0.01 compared with unit-wise folds.

**3.3 Baselines with the same history.**

The linear baselines are ridge regressions with 1 s of history (8 raised-cosine basis functions per input), regularisation chosen on the validation block. Inputs: single units, region counts, stimulus and behaviour, and their combinations.

| Broadband / 0.5–4 / 4–8 Hz test R² | Median over 89 sessions |
| --- | --- |
| CNN-LSTM | 0.631 / 0.380 / 0.320 |
| Ridge: units + stimulus + behaviour | 0.600 / 0.339 / 0.271 |
| Ridge: units, 1 s history | 0.598 / 0.321 / 0.266 |
| Ridge: region counts + stimulus + behaviour | 0.556 / 0.299 / 0.260 |
| Ridge: region counts | 0.499 / 0.170 / 0.136 |
| Ridge: units, current bin only | 0.201 / 0.040 / 0.034 |

The CNN-LSTM beats the strongest ridge by +0.024, +0.020 and +0.036 (79, 57 and 68 of 89 sessions; p < 0.001). R² is 1 − SS_res/SS_tot on the held-out block, negative values kept. Early stopping is on the validation loss. **[Authors]** Parameter counts, repeated seeds and the CNN-only model are still to do (Pending).

**3.4 Transfer is not generalisation.**

We now call it within-session transfer throughout and state that the models are session-specific (each session has different units). **[Authors]** Uncertainty and the spontaneous-epoch result wait on the scaling analysis (Pending).

**3.5 Correlated inputs and confounds; region removal.**

We removed each region at inference and after retraining, with and without covariates (stimulus onsets per image, change and omission, running, pupil, licks, rewards and summed population activity):

| R² lost, broadband (89 sessions) | At inference | Retrained | Retrained with covariates |
| --- | --- | --- | --- |
| VISp | 0.353 | 0.084 | 0.076 |
| Other visual cortex | 0.108 | 0.029 | 0.025 |
| Hippocampus | 0.024 | 0.024 | 0.016 |
| Thalamus | 0.004 | 0.002 | 0.002 |
| Midbrain | 0.007 | 0.004 | 0.001 |

Visual cortex and hippocampus carry information beyond stimulus timing, arousal and global activity. Thalamus and midbrain contribute at most 0.01; we have removed the claims about them. The language is predictive throughout, and sign is not interpreted as depolarisation or hyperpolarisation. **[Authors]** Rank stability across seeds and a second attribution method are Pending.

**3.7 Normalisation; validate on simulated data with known sources.**

The L1 normalisation is removed (Reviewer 2, point 2.3). We built the requested simulation: recorded spike trains of 3 sessions drive a synthetic LFP through 200 driver units with causal kernels (10–100 ms), under four generative models, plus noise. The CNN-LSTM is trained and attributed exactly as on the data. Attribution identifies the drivers (AUC 0.71–0.79) better than ridge weights and far better than spike–LFP correlation, spike-triggered LFP, population coupling or firing rate (Reviewer 2, point 2.1). Rankings with zero and training-mean IG baselines correlate at 0.94–0.95. **[Authors]** Fixed-size unit subsamples are Pending.

**3.9 Pseudoreplication.**

All comparisons in the revision are made per session (median over channels) and tested over sessions with paired Wilcoxon tests; decoders use mouse-grouped folds with within-session permutation nulls (50 shuffles). **[Authors]** Mixed models (sessions within mice), CIs and multiple-comparison correction are Pending.

**3.10 Clusters vs layers.**

Both decoders now use the same units and mouse-grouped folds with a permutation null: layer 0.393 (chance 0.25, null 0.253) and cluster 0.412 (chance 0.20, null 0.208). Firing rate alone decodes clusters at 0.251. Adding attribution to firing rate improves layer decoding by +0.067 (5 of 5 folds). **[Authors]** Decoding clusters within layers is Pending.

**3.12 Fig. 8 random forest.**

Refit with mouse-grouped cross-validation and held-out permutation importance: R² 0.302 ± 0.019 (permutation null −0.003). The VISp label dominates the importance (0.33), then firing rate (0.06) and distance from VISp. **[Authors]** Separating predictors coupled to attribution (stLFP, coherence) from independent ones is Pending.

**3.13 Cell types.** VIP is now descriptive only, with the number of mice and sessions per optotagged group; waveform classes are "putative". The pyramidal-vs-interneuron decoding claim is removed: firing rate alone decodes it better (0.935) than attribution (0.687).

**Minor points.** **[Authors]** Text fixes, to confirm in the revision: R² equation (1); Fig. 6E/6F reference and neuron counts per panel (2, 3; Figure 6 was made in MATLAB outside the repository); CIs (4); CSD and layer 4 border procedure (5, 6); log-normality QQ plot (7); Fig. 7B is r = −0.53 over 5 areas, p = 0.36, so the claim must go (8); LGN terminology (9); code release (10); predictive verbs (11); language edit (12).

## Pending

Fifteen items still need an analysis or a decision before their responses can be written; two are running now.

| Comment | Needs | State |
| --- | --- | --- |
| R1 1.1 | Frequency split confirmed after retraining (0.5–8 Hz) | Running on all 89 sessions |
| R1 1.3, R3 3.1 | Which regions carry the trial-specific part | Running on one session; then all 89 |
| R3 3.4 | Spontaneous-epoch R² under per-epoch scaling | Code ready, not submitted |
| R1 1.7, R3 3.9 | Statistics: mixed models (sessions within mice), CIs, multiple-comparison correction | Not started |
| R3 3.10 | Cluster decoding within layers; cluster × layer table | Not started |
| R3 3.3 | Repeated seeds, parameter counts, CNN-only model | Not started |
| R3 3.5 | Rank stability across seeds; a second attribution method | Not started |
| R3 3.6 | Bands above 25 Hz: per-model null and threshold, or cut | Decision needed |
| R3 3.7 | Fixed-size unit subsamples | Not started |
| R3 3.8 | Firing-rate control and ablation: select on one block, test on another; matched controls | Code partly ready |
| R3 3.11 | Spike bleed-through: unit-to-contact distances; exclude nearby units | Analysis done (89 sessions); text to draft |
| R3 3.12 | Fig. 8: predictors coupled to attribution vs independent ones | Not started |
| R2 2.2 | Timing claims: attribution peak lags recover true latencies poorly in simulation (Spearman 0.11–0.34) | Decision: coarse early/late wording or drop |
| R2 2.4 | R² correction: drop or keep as sensitivity analysis | Decision needed |
| R1 1.5, 1.6, R3 3.13, minor | Text changes | With the text revision |
