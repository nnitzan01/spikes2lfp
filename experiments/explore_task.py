"""
Exploratory look at one session before the region-contribution analyses (Phase 2 F-H): the task,
behaviour and pupil quality, and the event-locked LFP, band power, firing rates and model predictions.

Figures go to plots/<session_id>/explore/:
  1_timeline.png    the whole session: stimulus blocks, the active epoch's train / val / test blocks,
                    running speed, pupil area (raw and cleaned), lick and reward rate, engagement, hits
  2_pupil.png       pupil quality: raw distribution, missing / outlier / stable fraction per minute,
                    gap lengths, a typical and the worst 5 min of the active epoch, raw vs cleaned
  3_behaviour.png   first-lick latency after changes; licks, running and pupil around each event type
  4_lfp_events.png  event-locked broadband LFP (z-scored with the training scaler) by channel, per event
                    type, active and passive (layer 4 marked)
  5_bands.png       event-locked band power at the layer 4 channel
  6_population.png  event-locked firing rate (spikes/s per unit) by region group
  7_model.png       event-locked target and seed-0 prediction (broadband, 0.5-4 Hz) on the test block,
                    three depths (skipped with --no-model)
and the numbers to variables/<session_id>/explore_task.npz (summarize explore_task pools them).

Events (onsets of the active-epoch presentations; the passive replay's are prefixed passive_):
  image           a repeated image: not a change, not omitted, not right after an omission
  change_hit      a change with a lick RESPONSE_WINDOW s after it; change_miss without (auto-rewarded
                  trials dropped when the AllenSDK trials table is available)
  omission        an omitted flash; post_omission the image after it
  image_familiar, image_novel   repeated images by is_image_novel (novel sessions only)
Behaviour:
  pupil     clean_pupil: blinks and non-positive values removed; outliers removed with PUPIL_PAD_S on
            each side: samples off a PUPIL_MEDIAN_S running median by more than PUPIL_OUTLIER_SD robust SD
            (median absolute deviation over PUPIL_SCALE_S, at least PUPIL_MIN_SD of the median area:
            single-frame tracking errors), and samples beyond PUPIL_RANGE_IQR interquartile ranges of the
            session's log area (runs of failed fits); gaps up to PUPIL_MAX_GAP_S interpolated; kept only in
            stable stretches (>= PUPIL_STABLE_FRACTION valid samples in a centred PUPIL_STABLE_WINDOW_S window)
  running   1 s smoothed speed > RUNNING_CM_S; stationary < STATIONARY_CM_S
  engaged   reward rate over a centred ENGAGED_WINDOW_S window > ENGAGED_REWARDS_PER_MIN

usage (from the repo root):
    python -m experiments.explore_task <session_dir> [--no-model]

Needs the 'train' stage for the model figure.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import binary_dilation, uniform_filter1d  # noqa: E402

import config  # noqa: E402
import paths  # noqa: E402
from analysis.stimulus_clock import onset_bins  # noqa: E402
from experiments.pool_sessions import ALIGNMENT_TABLE  # noqa: E402
from experiments.region_ablation import GLOBAL_REGIONS, region_masks  # noqa: E402
from experiments.stimulus_clock import predict  # noqa: E402
from run_session import SessionPipeline, log, save_npz  # noqa: E402
from splits import chunk_bins  # noqa: E402

RESPONSE_WINDOW = (0.15, 0.75)      # s after a change
EVENT_WINDOW = (-0.25, 0.75)        # s around onsets
MAX_EVENTS = 1000                   # events averaged per type (random subset beyond that)
PUPIL_MEDIAN_S, PUPIL_SCALE_S, PUPIL_OUTLIER_SD, PUPIL_MIN_SD = 1.0, 30.0, 5.0, 0.01
PUPIL_RANGE_IQR, PUPIL_PAD_S = 3.0, 0.1
PUPIL_MAX_GAP_S = 1.0
PUPIL_STABLE_WINDOW_S, PUPIL_STABLE_FRACTION = 30.0, 0.8
RUNNING_CM_S, STATIONARY_CM_S = 5.0, 1.0
ENGAGED_WINDOW_S, ENGAGED_REWARDS_PER_MIN = 120.0, 2.0
ACTIVE_TYPES = ['image', 'change_hit', 'change_miss', 'omission', 'post_omission', 'image_familiar', 'image_novel']
PASSIVE_TYPES = ['passive_image', 'passive_change', 'passive_omission']
MODEL_TARGETS = [0, 1]


# ---- behaviour -------------------------------------------------------------------------------

def runs(mask):
    """(starts, lengths) of the runs of True in a boolean array."""
    d = np.diff(np.r_[0, np.asarray(mask, int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return starts, ends - starts


def clean_pupil(t, area, blink=None):
    """
    Cleaned pupil area (see the module docstring): dict of clean (NaN outside stable stretches and in long
    gaps), missing (NaN, non-positive or blink), outlier and stable (boolean per sample).
    """
    t, x = np.asarray(t, float), np.asarray(area, float).copy()
    missing = ~np.isfinite(x) | (x <= 0)
    if blink is not None:
        missing |= np.asarray(blink, bool)
    x[missing] = np.nan
    dt = np.nanmedian(np.diff(t)) if len(t) > 1 else 1.0
    n = lambda s: max(1, int(round(s / dt)))
    s = pd.Series(x)
    med = s.rolling(n(PUPIL_MEDIAN_S), center=True, min_periods=1).median()
    dev = (s - med).abs()
    sd = np.maximum(1.4826 * dev.rolling(n(PUPIL_SCALE_S), center=True, min_periods=1).median(), PUPIL_MIN_SD * med)
    log_x = np.log(x)
    q1, q3 = np.nanpercentile(log_x, [25, 75]) if np.isfinite(log_x).any() else (np.nan, np.nan)
    with np.errstate(invalid='ignore'):
        outlier = ((dev > PUPIL_OUTLIER_SD * sd).to_numpy() | (log_x < q1 - PUPIL_RANGE_IQR * (q3 - q1))
                   | (log_x > q3 + PUPIL_RANGE_IQR * (q3 - q1))) & ~missing
    bad = binary_dilation(missing | outlier, np.ones(2 * n(PUPIL_PAD_S) + 1, bool))
    x[bad] = np.nan
    stable = pd.Series((~bad).astype(float)).rolling(n(PUPIL_STABLE_WINDOW_S), center=True, min_periods=1).mean().to_numpy() \
        >= PUPIL_STABLE_FRACTION
    clean = pd.Series(x).interpolate(limit_area='inside').to_numpy().copy()
    for start, length in zip(*runs(np.isnan(x))):
        if length > n(PUPIL_MAX_GAP_S):
            clean[start:start + length] = np.nan
    clean[~stable] = np.nan
    return dict(clean=clean, missing=missing, outlier=outlier, stable=stable)


def event_rate(times, grid, window_s):
    """Events per second in a centred window_s window, at the grid times (regular grid)."""
    dt = grid[1] - grid[0]
    counts = np.histogram(np.asarray(times, float), bins=np.append(grid, grid[-1] + dt))[0].astype(float)
    return uniform_filter1d(counts, max(1, int(round(window_s / dt))), mode='constant') / dt


def on_grid(t, x, grid):
    """x (sampled at t) at the grid times; NaN where x is NaN at the nearest sample or outside t."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    i = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    out = x[i].copy()
    out[(grid < t[0]) | (grid > t[-1])] = np.nan
    return out


# ---- events ----------------------------------------------------------------------------------

def task_events(presentations, licks, auto_rewarded=(), prefix=''):
    """
    {event type: onset times (s)} of a presentation table (see the module docstring), and the first-lick
    latency after each change (s; inf if none). With a prefix (passive replay) changes are not split by licks.
    """
    p = presentations.sort_values('start_time').reset_index(drop=True)
    onset = p.start_time.to_numpy(float)
    omitted = (p.image_name.astype(str) == 'omitted').to_numpy()
    if 'omitted' in p:
        omitted |= p['omitted'].fillna(False).to_numpy(bool)
    change = p.is_change.fillna(False).to_numpy(bool) & ~omitted
    after_omission = np.r_[False, omitted[:-1]] & ~omitted
    image = ~omitted & ~change & ~after_omission
    licks = np.sort(np.asarray(licks, float))
    i = np.searchsorted(licks, onset)
    latency = np.where(i < len(licks), np.append(licks, np.inf)[i] - onset, np.inf)
    if prefix:
        events = {'image': onset[image], 'change': onset[change], 'omission': onset[omitted]}
        return {prefix + k: v for k, v in events.items()}, latency[change]
    responded = np.searchsorted(licks, onset + RESPONSE_WINDOW[0]) < np.searchsorted(licks, onset + RESPONSE_WINDOW[1])
    auto = np.zeros(len(p), bool)
    for a in np.asarray(auto_rewarded, float):
        auto |= np.abs(onset - a) < 0.1
    events = dict(image=onset[image], change_hit=onset[change & responded & ~auto],
                  change_miss=onset[change & ~responded & ~auto], omission=onset[omitted],
                  post_omission=onset[after_omission & ~change])
    if 'is_image_novel' in p and p['is_image_novel'].fillna(False).astype(bool).any():
        novel = p['is_image_novel'].fillna(False).to_numpy(bool)
        events['image_familiar'], events['image_novel'] = onset[image & ~novel], onset[image & novel]
    return events, latency[change & ~auto]


def event_average(x, onsets, pre, post, rng, max_events=MAX_EVENTS):
    """Mean of x (bins x ...; NaN ignored) from pre bins before to post bins after each onset bin; (mean, n events)."""
    onsets = np.asarray(onsets, int)
    onsets = onsets[(onsets - pre >= 0) & (onsets + post <= len(x))]
    if len(onsets) > max_events:
        onsets = np.sort(rng.choice(onsets, max_events, replace=False))
    total, count = np.zeros((pre + post,) + x.shape[1:]), np.zeros((pre + post,) + x.shape[1:])
    for chunk in np.array_split(onsets, max(1, len(onsets) // 100)):
        if len(chunk):
            window = x[chunk[:, None] + np.arange(-pre, post)]
            total += np.nansum(window, axis=0)
            count += np.isfinite(window).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / count, np.nan), len(onsets)


# ---- session -------------------------------------------------------------------------------

def auto_rewarded_changes(session):
    """Change times of auto-rewarded trials from the AllenSDK trials table; empty if not available."""
    try:
        trials = session.session.trials
        return trials.loc[trials.auto_rewarded.fillna(False).astype(bool), 'change_time_no_display_delay'].dropna().to_numpy()
    except Exception:       # fake sessions and tables without these columns
        return np.array([])


def passive_presentations(session):
    if hasattr(session, 'passive'):
        return session.passive
    stim = session.stimulus_presentation
    start, stop = session.passive_times
    return stim[~stim.active & (stim.start_time >= start) & (stim.start_time < stop)]


def layer4_channel(session_id, n_ch):
    """(L4 channel index, (lower, upper) border indices) from the alignment table; the middle channel if absent."""
    try:
        row = pd.read_csv(ALIGNMENT_TABLE).set_index('session').loc[session_id]
        borders = sorted([int(row['lower_border_index']), int(row['upper_border_index'])])
        if borders[1] < n_ch:
            return int(round(np.mean(borders))), borders
    except (KeyError, FileNotFoundError):
        pass
    return n_ch // 2, None


def epoch_signals(pipeline, start, stop, counts, lfp, pupil, rng, events, masks, test=None):
    """Event-locked averages of LFP, region firing rates, running, pupil and licks for one epoch."""
    s = pipeline.session
    t = counts.timestamps
    pre, post = int(round(-EVENT_WINDOW[0] / config.BIN_SIZE)), int(round(EVENT_WINDOW[1] / config.BIN_SIZE))
    centres = t + config.BIN_SIZE / 2
    rates = np.stack([counts.spkMat[:, masks[g]].mean(1) if masks[g].any() else np.full(len(t), np.nan)
                      for g in GLOBAL_REGIONS], 1) / config.BIN_SIZE
    behaviour = np.stack([on_grid(s.running_speed.timestamps, s.running_speed.speed, centres),
                          on_grid(pupil['t'], pupil['z'], centres),
                          np.histogram(s.licks.timestamps, bins=np.append(t, t[-1] + config.BIN_SIZE))[0] / config.BIN_SIZE], 1)
    out = {}
    for name, times in events.items():
        b = onset_bins(times, t, config.BIN_SIZE)
        b = b[(b >= 0) & (b < len(t))]
        if test is not None:
            out[f'{name}_n_test'] = int(np.sum((b - pre >= test[0]) & (b + post <= test[-1] + 1)))
        out[f'{name}_lfp'], out[f'{name}_n'] = event_average(lfp, b, pre, post, rng)
        out[f'{name}_pop'] = event_average(rates, b, pre, post, rng)[0]
        out[f'{name}_behaviour'] = event_average(behaviour, b, pre, post, rng)[0]
    return out


def model_averages(pipeline, events, pre, post, rng):
    """Event-locked target and prediction on the test block: {type: (target, prediction)} (lags x channels x MODEL_TARGETS)."""
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    test = chunk_bins(pipeline.split['test'])
    n_ch = Y.shape[1]
    yhat = np.full((len(X), n_ch, len(MODEL_TARGETS)), np.nan, np.float32)
    for c in range(n_ch):
        for j, band in enumerate(MODEL_TARGETS):
            yhat[:, c, j] = predict(pipeline.models[c, band], X, np.arange(len(X) // config.SEQLENGTH))
    t = pipeline.active_spikes.timestamps
    out = {}
    for name, times in events.items():
        b = onset_bins(times, t, config.BIN_SIZE)
        b = b[(b - pre >= test[0]) & (b + post <= test[-1] + 1)]
        out[name] = (event_average(Y[:, :, MODEL_TARGETS], b, pre, post, rng)[0],
                     event_average(yhat, b, pre, post, rng)[0], len(b))
    return out


# ---- figures -------------------------------------------------------------------------------

def contiguous_spans(bins, timestamps):
    """(start, stop) times of the contiguous runs of a bin index array."""
    bins = np.sort(bins)
    if not len(bins):
        return []
    breaks = np.flatnonzero(np.diff(bins) > 1)
    starts, stops = bins[np.r_[0, breaks + 1]], bins[np.r_[breaks, len(bins) - 1]]
    return [(timestamps[a], timestamps[b]) for a, b in zip(starts, stops)]


def figure_timeline(f, s, pipeline, grid, beh, events):
    fig, ax = plt.subplots(6, 1, figsize=(14, 12), sharex=True, gridspec_kw=dict(height_ratios=[0.6, 1, 1, 1, 1, 1]))
    m = grid / 60
    blocks = [('active', s.active_times, 'tab:blue'), ('spontaneous', s.spontaneous_times, 'tab:gray'),
              ('gabor', getattr(s, 'gabor_times', None), 'tab:olive'), ('flash', getattr(s, 'flash_times', None), 'tab:purple'),
              ('passive', s.passive_times, 'tab:orange')]
    for name, times, color in blocks:
        if times is not None:
            ax[0].axvspan(times[0] / 60, times[1] / 60, ymin=0.5, color=color, alpha=0.4)
            ax[0].text(np.mean(times) / 60, 0.75, name, ha='center', va='center', fontsize=8,
                       transform=ax[0].get_xaxis_transform())
    ts = pipeline.active_counts.timestamps
    for part, color in (('train', 'tab:green'), ('val', 'gold'), ('test', 'tab:red')):
        for a, b in contiguous_spans(chunk_bins(pipeline.split[part]), ts):
            ax[0].axvspan(a / 60, b / 60, ymax=0.45, color=color, alpha=0.6)
        ax[0].plot([], [], color=color, lw=6, label=part)
    ax[0].legend(loc='lower right', ncol=3, fontsize=7)
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks([])
    ax[0].set_title(f'session {pipeline.session_id}: blocks (top) and active-epoch split (bottom)')
    ax[1].plot(m, beh['running'], lw=0.5, color='k')
    ax[1].axhline(RUNNING_CM_S, color='tab:red', lw=0.5)
    ax[1].set_ylabel('running (cm/s)')
    ax[2].plot(m, beh['pupil_raw'], lw=0.4, color='0.7', label='raw')
    ax[2].plot(m, beh['pupil_clean'], lw=0.5, color='k', label='cleaned (stable only)')
    ax[2].legend(loc='upper right', fontsize=7)
    ax[2].set_ylabel('pupil area')
    ax[3].plot(m, beh['lick_rate'], lw=0.5, color='tab:blue')
    ax[3].set_ylabel('licks/s (10 s)')
    ax[4].plot(m, beh['reward_rate'], lw=0.8, color='tab:green')
    ax[4].axhline(ENGAGED_REWARDS_PER_MIN, color='tab:red', lw=0.5)
    ax[4].fill_between(m, 0, beh['reward_rate'].max() if np.isfinite(beh['reward_rate']).any() else 1,
                       where=beh['engaged'], color='tab:green', alpha=0.1, lw=0)
    ax[4].set_ylabel('rewards/min\n(shaded: engaged)')
    hits, misses = events.get('change_hit', []), events.get('change_miss', [])
    times = np.sort(np.r_[hits, misses])
    if len(times):
        outcome = np.isin(times, hits).astype(float)
        ax[5].plot(times / 60, pd.Series(outcome).rolling(10, center=True, min_periods=3).mean(), color='k')
        ax[5].plot(times / 60, outcome, '|', color='tab:green', ms=4)
    ax[5].set_ylabel('hit rate\n(10 changes)')
    ax[5].set_xlabel('time (min)')
    fig.tight_layout()
    fig.savefig(f, dpi=110)
    plt.close(fig)


def figure_pupil(f, eye_t, raw, pupil, active_times):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    finite = raw[np.isfinite(raw) & (raw > 0)]
    if len(finite):
        bins = np.linspace(*np.percentile(finite, [0, 100]), 100)
        ax[0, 0].hist(finite, bins, color='0.6', label='all valid')
        ax[0, 0].hist(raw[pupil['outlier']], bins, color='tab:red', label='outliers')
        ax[0, 0].set_yscale('log')
        ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_xlabel('pupil area (raw)')
    minute = (eye_t // 60).astype(int)
    per = pd.DataFrame(dict(minute=minute, missing=pupil['missing'], outlier=pupil['outlier'], stable=pupil['stable'])).groupby('minute').mean()
    for k, color in (('missing', 'k'), ('outlier', 'tab:red'), ('stable', 'tab:green')):
        ax[0, 1].plot(per.index, per[k], color=color, label=k)
    ax[0, 1].axvspan(active_times[0] / 60, active_times[1] / 60, color='tab:blue', alpha=0.1, label='active')
    ax[0, 1].set_xlabel('time (min)')
    ax[0, 1].set_ylabel('fraction of samples')
    ax[0, 1].legend(fontsize=7)
    dt = np.nanmedian(np.diff(eye_t))
    _, lengths = runs(np.isnan(np.where(pupil['missing'] | pupil['outlier'], np.nan, 1.0)))
    if len(lengths):
        ax[0, 2].hist(lengths * dt, np.logspace(np.log10(dt), np.log10(max(lengths.max() * dt, 2 * dt)), 40), color='0.4')
        ax[0, 2].set_xscale('log')
        ax[0, 2].set_yscale('log')
        ax[0, 2].axvline(PUPIL_MAX_GAP_S, color='tab:red', lw=0.8, label=f'interpolated up to {PUPIL_MAX_GAP_S} s')
        ax[0, 2].legend(fontsize=7)
    ax[0, 2].set_xlabel('gap length (s)')
    # a typical and the worst 5 min of the active epoch, by fraction of bad samples
    in_active = (eye_t >= active_times[0]) & (eye_t < active_times[1])
    bad = pd.Series((pupil['missing'] | pupil['outlier']).astype(float))
    win = max(1, int(round(min(300, np.diff(active_times)[0] / 2) / dt)))
    frac = bad.rolling(win, min_periods=win).mean().to_numpy().copy()
    frac[~in_active] = np.nan
    if np.isfinite(frac).any():
        valid = np.flatnonzero(np.isfinite(frac))
        typical = valid[np.argmin(np.abs(frac[valid] - np.nanmedian(frac)))]
        worst = valid[np.nanargmax(frac[valid])]
        for a, (end, label) in zip((ax[1, 0], ax[1, 1]), ((typical, 'typical'), (worst, 'worst'))):
            sl = slice(max(0, end - win), end)
            a.plot(eye_t[sl] / 60, raw[sl], color='0.7', lw=0.5, label='raw')
            out = np.flatnonzero(pupil['outlier'][sl]) + sl.start
            a.plot(eye_t[out] / 60, raw[out], '.', color='tab:red', ms=3, label='outlier')
            a.plot(eye_t[sl] / 60, pupil['clean'][sl], color='k', lw=0.8, label='cleaned')
            a.set_title(f'{label} {win * dt / 60:.0f} min of the active epoch ({frac[end]:.0%} bad samples)', fontsize=9)
            a.set_xlabel('time (min)')
            a.legend(fontsize=7)
    act = pupil['stable'][in_active]
    ax[1, 2].axis('off')
    ax[1, 2].text(0, 0.9, '\n'.join([
        f'samples: {len(raw)} at {1 / dt:.0f} Hz',
        f'missing / blink: {pupil["missing"].mean():.1%}',
        f'outliers: {pupil["outlier"].mean():.1%}',
        f'stable, whole session: {pupil["stable"].mean():.1%}',
        f'stable, active epoch: {act.mean() if len(act) else np.nan:.1%}',
        '', f'outlier: > {PUPIL_OUTLIER_SD:g} robust SD from a {PUPIL_MEDIAN_S:g} s running median,',
        f'  or > {PUPIL_RANGE_IQR:g} IQR outside the log-area quartiles',
        f'stable: >= {PUPIL_STABLE_FRACTION:.0%} valid in {PUPIL_STABLE_WINDOW_S:.0f} s'
    ]), va='top', fontsize=9, family='monospace')
    fig.tight_layout()
    fig.savefig(f, dpi=110)
    plt.close(fig)


def figure_behaviour(f, latency, sig, lags, summary):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    finite = latency[np.isfinite(latency) & (latency < 2)]
    ax[0, 0].hist(finite, np.arange(0, 2, 0.025), color='0.4')
    ax[0, 0].axvspan(*RESPONSE_WINDOW, color='tab:green', alpha=0.15, label='response window')
    ax[0, 0].set_xlabel('first lick after change (s)')
    ax[0, 0].legend(fontsize=7)
    types = [k for k in ACTIVE_TYPES + PASSIVE_TYPES if f'{k}_behaviour' in sig and sig[f'{k}_n'] > 0]
    for j, (a, label) in enumerate(zip((ax[0, 1], ax[1, 0], ax[1, 1]), ('running (cm/s)', 'pupil (z, cleaned)', 'licks/s (100 ms)'))):
        for k in types:
            y = sig[f'{k}_behaviour'][:, j]
            if j == 2:
                y = uniform_filter1d(y, 25)                 # 100 ms
            a.plot(lags, y, label=f'{k} ({sig[f"{k}_n"]})', ls='--' if k.startswith('passive') else '-')
        a.axvline(0, color='k', lw=0.5)
        a.set_xlabel('time from onset (ms)')
        a.set_ylabel(label)
    ax[0, 1].legend(fontsize=6)
    fig.suptitle(', '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in summary.items()), fontsize=8)
    fig.tight_layout()
    fig.savefig(f, dpi=110)
    plt.close(fig)


def figure_lfp(f, sig, lags, l4, borders, depth_note):
    types = [k for k in ACTIVE_TYPES + PASSIVE_TYPES if f'{k}_lfp' in sig and sig[f'{k}_n'] > 0]
    ncol = 4
    fig, ax = plt.subplots(int(np.ceil(len(types) / ncol)), ncol, figsize=(16, 3.6 * np.ceil(len(types) / ncol)), squeeze=False,
                           sharex=True, sharey=True, layout='constrained')
    vmax = np.nanmax([np.nanpercentile(np.abs(sig[f'{k}_lfp'][..., 0]), 99) for k in types]) if types else 1
    for a, k in zip(ax.flat, types):
        im = a.imshow(sig[f'{k}_lfp'][..., 0].T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      extent=[lags[0], lags[-1], -0.5, sig[f'{k}_lfp'].shape[1] - 0.5])
        if borders:
            for b in borders:
                a.axhline(b, color='k', lw=0.6, ls='--')
        a.axvline(0, color='k', lw=0.5)
        a.set_title(f'{k} (n {sig[f"{k}_n"]})', fontsize=9)
    for a in ax[-1]:
        a.set_xlabel('ms')
    for a in ax[:, 0]:
        a.set_ylabel('channel')
    for a in list(ax.flat)[len(types):]:
        a.axis('off')
    if types:
        fig.colorbar(im, ax=ax, shrink=0.6, label='broadband LFP (z, training scaler)')
    fig.suptitle(f'event-locked broadband LFP; dashed: layer 4 borders; {depth_note}', fontsize=9)
    fig.savefig(f, dpi=110)
    plt.close(fig)


def figure_bands(f, sig, lags, l4):
    types = [k for k in ('image', 'change_hit', 'change_miss', 'omission', 'passive_image') if f'{k}_lfp' in sig and sig[f'{k}_n'] > 0]
    names = ['broadband'] + [f'{lo:g}-{hi:g} Hz' for lo, hi in config.BANDS]
    fig, ax = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for t, a in enumerate(ax.flat[:len(names)]):
        for k in types:
            a.plot(lags, sig[f'{k}_lfp'][:, l4, t], label=k, ls='--' if k.startswith('passive') else '-')
        a.axvline(0, color='k', lw=0.5)
        a.set_title(names[t], fontsize=9)
    ax[0, 0].legend(fontsize=7)
    for a in ax[-1]:
        a.set_xlabel('time from onset (ms)')
    fig.suptitle(f'event-locked targets at channel {l4} (layer 4), z (training scaler)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f, dpi=110)
    plt.close(fig)


def figure_population(f, sig, lags, n_units):
    types = [k for k in ACTIVE_TYPES + PASSIVE_TYPES if f'{k}_pop' in sig and sig[f'{k}_n'] > 0]
    fig, ax = plt.subplots(1, len(GLOBAL_REGIONS), figsize=(18, 4), sharex=True)
    for g, (a, region) in enumerate(zip(ax, GLOBAL_REGIONS)):
        for k in types:
            a.plot(lags, uniform_filter1d(sig[f'{k}_pop'][:, g], 5), label=k, ls='--' if k.startswith('passive') else '-')
        a.axvline(0, color='k', lw=0.5)
        a.set_title(f'{region} ({n_units[g]} units)', fontsize=9)
        a.set_xlabel('ms')
    ax[0].set_ylabel('spikes/s per unit (20 ms)')
    ax[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f, dpi=110)
    plt.close(fig)


def figure_model(f, model, lags, channels):
    types = [k for k in ('image', 'change_hit', 'change_miss', 'omission') if k in model and model[k][2] > 0]
    fig, ax = plt.subplots(len(MODEL_TARGETS), len(channels), figsize=(5 * len(channels), 4 * len(MODEL_TARGETS)), squeeze=False)
    names = ['broadband'] + [f'{lo:g}-{hi:g} Hz' for lo, hi in config.BANDS]
    for j, band in enumerate(MODEL_TARGETS):
        for i, (c, label) in enumerate(channels):
            a = ax[j, i]
            for n, k in enumerate(types):
                target, pred, count = model[k]
                color = f'C{n}'
                a.plot(lags, target[:, c, j], color=color, label=f'{k} ({count})')
                a.plot(lags, pred[:, c, j], color=color, ls='--')
            a.axvline(0, color='k', lw=0.5)
            a.set_title(f'{names[band]}, channel {c} ({label})', fontsize=9)
            a.set_xlabel('ms')
    ax[0, 0].legend(fontsize=7, title='solid target, dashed prediction', title_fontsize=7)
    fig.suptitle('test block: event-locked target and seed-0 model prediction (z)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f, dpi=110)
    plt.close(fig)


# ---- run -----------------------------------------------------------------------------------

def run(pipeline, with_model=True, seed=config.TORCH_SEED):
    rng = np.random.default_rng(seed)
    s = pipeline.session
    fig_dir = pipeline.dirs.plots / 'explore'
    fig_dir.mkdir(parents=True, exist_ok=True)
    pre, post = int(round(-EVENT_WINDOW[0] / config.BIN_SIZE)), int(round(EVENT_WINDOW[1] / config.BIN_SIZE))
    lags = (np.arange(-pre, post) + 0.5) * config.BIN_SIZE * 1e3

    # behaviour on a 0.5 s grid over the whole session
    end = max(s.passive_times[1], s.active_times[1], *(getattr(s, n)[1] for n in ('flash_times', 'gabor_times') if hasattr(s, n)))
    grid = np.arange(0, end, 0.5)
    run_speed = uniform_filter1d(np.asarray(s.running_speed.speed, float), max(1, int(round(1 / np.median(np.diff(s.running_speed.timestamps))))))
    eye = s.eye_tracking
    if eye is not None and len(eye):
        eye_t, raw = eye.timestamps.to_numpy(float), eye.pupil_area.to_numpy(float)
        pupil = clean_pupil(eye_t, raw, eye.likely_blink if 'likely_blink' in eye else None)
    else:
        eye_t, raw = np.array([0.0, 1.0]), np.full(2, np.nan)
        pupil = dict(clean=raw.copy(), missing=np.ones(2, bool), outlier=np.zeros(2, bool), stable=np.zeros(2, bool))
    mu, sd = np.nanmean(pupil['clean']), np.nanstd(pupil['clean'])
    pupil_z = dict(t=eye_t, z=(pupil['clean'] - mu) / sd if sd > 0 else pupil['clean'] * np.nan)
    reward_rate = event_rate(s.rewards.timestamps, grid, ENGAGED_WINDOW_S) * 60
    beh = dict(running=on_grid(s.running_speed.timestamps, run_speed, grid), pupil_raw=on_grid(eye_t, raw, grid),
               pupil_clean=on_grid(eye_t, pupil['clean'], grid), lick_rate=event_rate(s.licks.timestamps, grid, 10.0),
               reward_rate=reward_rate, engaged=reward_rate > ENGAGED_REWARDS_PER_MIN)

    # events
    events, latency = task_events(s.active, s.licks.timestamps, auto_rewarded_changes(s))
    passive, _ = task_events(passive_presentations(s), s.licks.timestamps, prefix='passive_')

    # event-locked signals, active (with test-block counts) and passive
    masks = region_masks(np.asarray(s.units.structure_acronym))
    test = chunk_bins(pipeline.split['test'])
    sig = epoch_signals(pipeline, *s.active_times, pipeline.active_counts, pipeline.active_lfp.lfpMat, pupil_z, rng,
                        events, masks, test)
    p_counts = pipeline.spike_counts(*s.passive_times)
    p_lfp = pipeline.target_scaler.transform(pipeline.lfp_targets(*s.passive_times, p_counts.spkMat.shape[0]).lfpMat)
    sig.update(epoch_signals(pipeline, *s.passive_times, p_counts, p_lfp, pupil_z, rng, passive, masks))
    log('events: ' + ', '.join(f'{k} {sig[f"{k}_n"]}' for k in list(events) + list(passive)))

    # per-session numbers
    in_active = (grid >= s.active_times[0]) & (grid < s.active_times[1])
    ts = pipeline.active_counts.timestamps
    in_test = (grid >= ts[test[0]]) & (grid <= ts[test[-1]])
    eye_active = (eye_t >= s.active_times[0]) & (eye_t < s.active_times[1])
    n_hit, n_miss = len(events['change_hit']), len(events['change_miss'])
    summary = dict(hit_rate=n_hit / max(1, n_hit + n_miss), lick_latency_median=float(np.median(latency[np.isfinite(latency)])) if np.isfinite(latency).any() else np.nan,
                   pupil_missing=float(pupil['missing'].mean()), pupil_outlier=float(pupil['outlier'].mean()),
                   pupil_stable_active=float(pupil['stable'][eye_active].mean()) if eye_active.any() else np.nan,
                   running_active=float(np.nanmean(beh['running'][in_active] > RUNNING_CM_S)),
                   stationary_active=float(np.nanmean(beh['running'][in_active] < STATIONARY_CM_S)),
                   engaged_active=float(beh['engaged'][in_active].mean()), engaged_test=float(beh['engaged'][in_test].mean()),
                   active_minutes=float(np.diff(s.active_times)[0] / 60))
    n_units = [int(masks[g].sum()) for g in GLOBAL_REGIONS]
    log('summary: ' + ', '.join(f'{k} {v:.3f}' for k, v in summary.items()))

    n_ch = pipeline.active_lfp.lfpMat.shape[1]
    l4, borders = layer4_channel(pipeline.session_id, n_ch)
    ch = pipeline.active_lfp.channels
    depth_note = ''
    if 'probe_vertical_position' in ch and len(ch) > 1:
        v = np.asarray(ch['probe_vertical_position'], float)
        depth_note = f'channel 0 at {v[0]:.0f} um, channel {len(v) - 1} at {v[-1]:.0f} um (larger = more superficial)'

    figures = [('1_timeline.png', lambda f: figure_timeline(f, s, pipeline, grid, beh, events)),
               ('2_pupil.png', lambda f: figure_pupil(f, eye_t, raw, pupil, s.active_times)),
               ('3_behaviour.png', lambda f: figure_behaviour(f, latency, sig, lags, summary)),
               ('4_lfp_events.png', lambda f: figure_lfp(f, sig, lags, l4, borders, depth_note)),
               ('5_bands.png', lambda f: figure_bands(f, sig, lags, l4)),
               ('6_population.png', lambda f: figure_population(f, sig, lags, n_units))]
    model = {}
    if with_model:
        model = model_averages(pipeline, events, pre, post, rng)
        picks = [(min(n_ch - 1, l4 + 4), 'above L4'), (l4, 'L4'), (max(0, l4 - 4), 'below L4')]
        figures.append(('7_model.png', lambda f: figure_model(f, model, lags, picks)))
    for name, draw in figures:
        try:
            draw(fig_dir / name)
        except Exception as e:      # one broken figure should not cost the others
            log(f'figure {name} failed: {e!r}')
            plt.close('all')

    types = [k for k in ACTIVE_TYPES + PASSIVE_TYPES if f'{k}_n' in sig]
    shape = sig[f'{types[0]}_lfp'].shape
    out = dict(event_types=np.array(types), eta_lags_ms=lags, region_groups=np.array(list(GLOBAL_REGIONS)),
               n_units=np.array(n_units), layer4_channel=l4,
               n_events=np.array([sig[f'{k}_n'] for k in types]),
               n_events_test=np.array([sig.get(f'{k}_n_test', -1) for k in types]),
               eta_lfp=np.stack([sig[f'{k}_lfp'] if sig[f'{k}_n'] else np.full(shape, np.nan) for k in types]),
               eta_pop=np.stack([sig[f'{k}_pop'] for k in types]),
               eta_behaviour=np.stack([sig[f'{k}_behaviour'] for k in types]), behaviour_names=np.array(['running', 'pupil_z', 'licks']),
               lick_latency=latency, **{k: np.float64(v) for k, v in summary.items()})
    if model:
        mt = [k for k in model]
        out.update(model_types=np.array(mt), model_targets=np.array(MODEL_TARGETS),
                   model_eta_target=np.stack([model[k][0] for k in mt]), model_eta_prediction=np.stack([model[k][1] for k in mt]),
                   model_n=np.array([model[k][2] for k in mt]))
    return out, fig_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--no-model', action='store_true', help='skip the model figure (no trained models needed)')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out, fig_dir = run(pipeline, with_model=not args.no_model)
    f = pipeline.dirs.variables / 'explore_task.npz'
    save_npz(f, **out)
    print('Saved', f, 'and figures in', fig_dir)


if __name__ == '__main__':
    main()
