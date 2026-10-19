"""
Plotting helpers for single-session results.

Every plotting function takes the same trailing arguments:
    show_plot  display the figure (otherwise it is closed after saving)
    save_fig   save it as output_dir / fig_name (format taken from the extension, PDF if none)
"""
import numpy as np
import torch
from scipy import signal
from pathlib import Path
import matplotlib.cm as cm
from scipy.signal import welch
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config

# Keep text editable in PDF/PS output (TrueType instead of Type 3 fonts)
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 10,
    'text.usetex': False
})


def save_plot(fig, output_dir, session_id=None, plot_name='default_name.pdf'):
    """Save fig to output_dir / plot_name, creating the directory. session_id is unused (kept for callers)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / plot_name
    fmt = file_path.suffix.lstrip('.') or 'pdf'
    fig.savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {file_path}")


def _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name):
    """Save (if requested) before showing or closing, so the saved file matches what is shown."""
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _band_title(bands, bandi):
    """Title for model target bandi: 0 is broadband, i > 0 is bands[i-1]."""
    return 'Broadband' if bandi == 0 else f'{bands[bandi-1][0]}-{bands[bandi-1][1]} Hz'


def _area_order(locs):
    """
    Order units by area for area-blocked matrix plots.

    Returns (sidx, areas, first, last, middle): the sorting index, the sorted unique areas, and the
    first/last/middle row of each area's block in the sorted order.
    """
    locs = np.asarray(locs)
    sidx = np.argsort(locs, kind='stable')
    sorted_locs = locs[sidx]
    areas = np.unique(sorted_locs)
    first = np.array([np.flatnonzero(sorted_locs == a)[0] for a in areas])
    last = np.array([np.flatnonzero(sorted_locs == a)[-1] for a in areas])
    middle = (first + (last - first) / 2).astype(int)
    return sidx, areas, first, last, middle


def _band_grid(n_panels, figsize):
    """Two-column grid with enough rows for n_panels; unused trailing axes are removed."""
    fig, ax = plt.subplots(int(np.ceil(n_panels / 2)), 2, figsize=figsize)
    for extra in ax.flat[n_panels:]:
        fig.delaxes(extra)
    return fig, ax


def plot_r2(scoresTest, channels, bands, clim=[-.4, .7], data_type='active', show_plot=True,
            save_fig=False, output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """
    Heatmap of test R2 for every channel and model target.

    scoresTest: (#channels, #bands+1); channels: channel table (rows in the same order)
    """
    num_channels = len(channels)
    assert scoresTest.shape == (num_channels, len(bands)+1)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(scoresTest, aspect='equal', cmap='viridis', vmin=clim[0], vmax=clim[1])
    ax.set_xlabel('Band')
    ax.set_ylabel('Channel')
    ax.set_title('Model score for all channels and bands on ' + data_type + ' data')
    ax.set_yticks(range(num_channels))
    ax.set_yticklabels(channels['dorsal_ventral_ccf_coordinate'], rotation=45)
    ax.set_xticks(range(len(bands)+1))
    ax.set_xticklabels(['Broadband'] + [str(b) for b in bands], rotation=45, ha='right')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    fig.colorbar(im).set_label('R2')
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_lfp_prediction(y, yHat, chan2use, bands, start_time=None, end_time=None, fs=1/config.BIN_SIZE, show_plot=True,
                        save_fig=False, output_dir=None, session_id=None, fig_name='lfp_prediction.pdf'):
    """
    Recorded vs predicted LFP of one channel for every model target.

    y: (time, channels, targets); yHat: same shape, or (time, targets) for chan2use only
    start_time, end_time: window to plot in seconds from the start of y (default: everything)
    """
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    y = y[:, chan2use, :]
    if yHat.ndim == 3:
        yHat = yHat[:, chan2use, :]

    start_idx = 0 if start_time is None else int(start_time * fs)
    end_idx = y.shape[0] if end_time is None else int(end_time * fs)
    t = np.arange(start_idx, end_idx) / fs

    fig, ax = _band_grid(len(bands)+1, figsize=(16, 16))
    for bandi in range(len(bands)+1):
        ax.flat[bandi].plot(t, y[start_idx:end_idx, bandi], 'k', label='LFP')
        ax.flat[bandi].plot(t, yHat[start_idx:end_idx, bandi], 'r', label='Model')
        ax.flat[bandi].set_title(f'Channel {chan2use} {_band_title(bands, bandi)}')
    ax.flat[0].legend(loc='upper right')
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)
    return fig


def plot_all_channel_loss(channels, bands, bandi, losses, show_plot=True,
                          save_fig=False, output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """
    Loss per epoch for every channel's model of one target, coloured by channel depth.

    losses: (#channels, #epochs, #bands+1)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    depth = channels['dorsal_ventral_ccf_coordinate']
    norm = mcolors.Normalize(vmin=depth.min(), vmax=depth.max())
    cmap = cm.viridis
    for chani in range(len(channels)):
        ax.plot(losses[chani, :, bandi], 'o-', alpha=0.5, color=cmap(norm(depth.iloc[chani])))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'Model loss for {_band_title(bands, bandi)} model')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label('Electrode Depth')
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_abs_error_change(y, yHat, session_obj, timestamps, chan2use, bands,
                          snippet_length=0.75, show_plot=True, save_fig=False, output_dir=None, session_id=None,
                          fig_name='default_name.pdf'):
    """
    Absolute prediction error around every active-task image presentation (-0.25 to +0.5 s),
    one row per presentation, with the mean error for change vs non-change images overlaid.

    y, yHat: (time, channels, targets); timestamps: (time,) in seconds
    """
    abs_error = np.abs(yHat[:, chan2use, :] - y[:, chan2use, :])    # (time, targets)
    start_times = session_obj.active.start_time.values
    keep = start_times < timestamps[-1] - snippet_length
    start_times = start_times[keep]
    change = np.array(session_obj.active.is_change.values, dtype=bool)[keep]

    nbins = int(snippet_length / config.BIN_SIZE)
    t = np.linspace(-0.25, 0.5, nbins)
    error_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    for i, st in enumerate(start_times):
        start = np.argmin(np.abs(timestamps - (st - .25)))
        error_snippets[i] = abs_error[start:start+nbins, :]

    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)), 2, figsize=(30, 30))
    for bandi in range(len(bands)+1):
        im = ax.flat[bandi].imshow(error_snippets[:, :, bandi], aspect='auto', cmap='bwr',
                                   extent=[-0.25, 0.5, 0, len(start_times)], vmin=0, vmax=2, alpha=0.9)
        ax2 = ax.flat[bandi].twinx()
        ax2.plot(t, np.mean(error_snippets[change, :, bandi], axis=0), 'm', label='change', linewidth=3)
        ax2.plot(t, np.mean(error_snippets[~change, :, bandi], axis=0), 'k', label='no change', linewidth=3)
        ax.flat[bandi].set_xlabel('Time from image onset (s)')
        ax.flat[bandi].set_title(_band_title(bands, bandi))
        if bandi == 0:
            ax.flat[bandi].set_ylabel('Presentation')
            ax2.legend(loc='upper left')
    # the spare last panel holds the shared colorbar
    ax.flat[-1].axis('off')
    fig.colorbar(im, ax=ax.flat[-1]).set_label('|error|')
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_psd(y, yHat, chani, fs=1/config.BIN_SIZE, show_plot=True, save_fig=False,
             output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """Welch PSD of the recorded and predicted broadband LFP of one channel (y, yHat: time x channels x targets)."""
    f, Pxx = welch(y[:, chani, 0], fs, nperseg=64)
    f, Pxx_hat = welch(yHat[:, chani, 0], fs, nperseg=64)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(f, np.log10(Pxx), 'r', label='LFP')
    ax.plot(f, np.log10(Pxx_hat), 'g', label='model')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log10(PSD)')
    ax.set_title('Power spectral density of LFP and model prediction')
    ax.set_xlim([0, 120])
    ax.legend()
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_attr_corrmat(attr, session_obj, show_plot=True, save_fig=False,
                      output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """Correlation between units' attribution time courses (attr: time x units), grouped by area."""
    sidx, areas, first, last, middle = _area_order(session_obj.units['structure_acronym'].values)
    corrmat = np.corrcoef(attr[:, sidx].T)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(corrmat, vmin=-.1, vmax=.1, aspect='auto', cmap='bwr')
    for boundary in last[:-1] + 0.5:       # lines between area blocks
        ax.axhline(y=boundary, color='white', linewidth=1, alpha=0.8)
        ax.axvline(x=boundary, color='white', linewidth=1, alpha=0.8)
    ax.set_xticks(middle)
    ax.set_xticklabels(areas, rotation=45, ha='right')
    ax.set_yticks(middle)
    ax.set_yticklabels(areas, rotation=45, ha='right')
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04).set_label('Correlation')
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_attr_density_plots(mean_attribution, session_obj, bands, area, show_plot=True, save_fig=False,
                            output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """
    Histograms of |mean attribution| over channels and units of one area, per model target.

    mean_attribution: (channels, targets, units); `area` is matched as a substring of the acronym.
    """
    in_area = session_obj.units['structure_acronym'].str.contains(area).values
    exp = np.ceil(np.log10(np.abs(np.median(mean_attribution))))   # x range up to the median's order of magnitude

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for bandi in range(len(bands)+1):
        values = np.abs(mean_attribution[:, bandi, in_area].flatten())
        ax.hist(values, bins=50, range=(0, 10 ** exp), histtype='step', label=_band_title(bands, bandi),
                weights=np.ones(len(values)) / len(values))
    ax.set_xlabel('Attribution')
    ax.set_ylabel('Probability')
    ax.set_xlim([- 10 ** (exp-2), 10 ** exp])
    ax.legend(loc='upper right', frameon=False)
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_mean_attr_areas(mean_attr_areas, lfp_obj, session_obj, bands, show_plot=True, save_fig=False,
                         output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """|Mean attribution| per area (rows, sorted) and channel (columns), one panel per model target."""
    num_channels = len(lfp_obj.channels)
    areas = sorted(session_obj.units['structure_acronym'].unique())

    fig, ax = _band_grid(len(bands)+1, figsize=(10, 20))
    for bandi in range(len(bands)+1):
        im = ax.flat[bandi].imshow(np.abs(mean_attr_areas[:, bandi, :]), aspect='auto', cmap='bwr')
        ax.flat[bandi].set_xticks(range(num_channels))
        ax.flat[bandi].set_xticklabels(lfp_obj.channels['dorsal_ventral_ccf_coordinate'].astype(int),
                                       rotation=45, ha='right')
        ax.flat[bandi].set_xlabel('dorsal ventral ccf coordinate')
        ax.flat[bandi].set_yticks(range(len(areas)))
        ax.flat[bandi].set_yticklabels(areas)
        ax.flat[bandi].set_ylabel('Area')
        ax.flat[bandi].set_title(_band_title(bands, bandi))
        fig.colorbar(im, ax=ax.flat[bandi])
    plt.tight_layout()
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_mean_attr_pyr_int(mean_attribution, session_obj, area, df, bands, chan2use, show_plot=True, save_fig=False,
                           output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """Boxplots of |mean attribution| of putative pyramidal cells vs interneurons (df.pyr) in one area and channel."""
    area_idx = session_obj.units.index[session_obj.units['structure_acronym'].str.contains(area)]
    pyr = np.intersect1d(df.unit_id[df.pyr == 1], area_idx)
    intn = np.intersect1d(df.unit_id[df.pyr == 0], area_idx)
    attr_pyr = mean_attribution[:, :, session_obj.units.index.isin(pyr)]
    attr_int = mean_attribution[:, :, session_obj.units.index.isin(intn)]

    fig, ax = _band_grid(len(bands)+1, figsize=(5, 8))
    for bandi in range(len(bands)+1):
        ax.flat[bandi].boxplot([np.abs(attr_pyr[chan2use, bandi, :]), np.abs(attr_int[chan2use, bandi, :])],
                               showfliers=False)
        ax.flat[bandi].set_title(_band_title(bands, bandi))
        ax.flat[bandi].set_xticklabels(['pyr', 'int'])
        ax.flat[bandi].set_ylabel('Mean attribution')
    plt.tight_layout()
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_attr_matrix(attr, lfp, timestamps, session_obj, time_win, plot_stim=False, bin_size=config.BIN_SIZE,
                     show_plot=True, save_fig=False, output_dir=None, session_id=None, fig_name='attr_matrix.pdf',
                     min_neurons=None):
    """
    LFP trace above the attribution raster (units grouped by area) for time_win = [start, stop] s.

    attr: (time, units) aligned with timestamps; min_neurons: only label areas with at least this
    many units; plot_stim: mark image onsets and offsets.
    """
    stim_st = session_obj.stimulus_presentation.start_time
    stim_st = stim_st[(stim_st > time_win[0]) & (stim_st < time_win[1])].values

    sidx, all_areas, all_first, all_last, _ = _area_order(session_obj.units['structure_acronym'].values)
    if min_neurons is not None:
        large = (all_last - all_first + 1) >= min_neurons
        print(f"Filtered out {np.sum(~large)} areas with <{min_neurons} neurons: {', '.join(all_areas[~large])}")
        areas, first, last = all_areas[large], all_first[large], all_last[large]
    else:
        areas, first, last = all_areas, all_first, all_last
    middle = (first + (last - first) / 2).astype(int)

    st = int((time_win[0] - timestamps[0]) / bin_size)
    en = int((time_win[1] - timestamps[0]) / bin_size)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[1, 4])
    ax1.plot(timestamps, lfp, 'k')
    ax1.set_xlim([time_win[0], time_win[1]])

    im = ax2.imshow(attr[st:en, sidx].T, aspect='auto', cmap='bwr', vmin=-0.0001, vmax=0.0001,
                    extent=[time_win[0], time_win[1], len(session_obj.units), 0])
    for boundary in last[:-1] + 0.5:       # lines between area blocks
        ax2.axhline(y=boundary, color='gray', linewidth=1, alpha=0.8, linestyle='--')
    if plot_stim:
        for onset in stim_st:
            ax2.axvline(onset, color='k', linestyle='--')
            ax2.axvline(onset + .25, color='k', linestyle='--')   # images are shown for 250 ms
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Unit')
    ax2.set_xlim([time_win[0], time_win[1]])
    ax2.set_yticks(middle)
    ax2.set_yticklabels(areas, rotation=45, ha='right')

    # give both panels the same width and put the colorbar beside them
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    pos1, pos2 = ax1.get_position(), ax2.get_position()
    ax1.set_position([pos1.x0, pos1.y0, pos2.width, pos1.height])
    cbar_ax = fig.add_axes([0.87, pos2.y0, 0.02, pos1.y0 + pos1.height - pos2.y0])
    fig.colorbar(im, cax=cbar_ax).set_label('Attribution')
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_attr_lfp_corr(attr, lfp, session_obj, sorting="max", bin_size=config.BIN_SIZE,
                       show_plot=True, save_fig=False, output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """
    Normalised cross-correlation between each unit's attribution and the LFP, within +-1 s lag.

    sorting='max': order units by mean correlation within +-200 lags of zero; otherwise by area.
    """
    sidx, areas, _, _, middle = _area_order(session_obj.units['structure_acronym'].values)
    lfp = lfp.reshape(-1)

    cross_corr = np.array([signal.correlate(lfp, attr[:, i], mode='same') for i in range(attr.shape[1])])
    corr_norm = cross_corr / (np.std(attr, axis=0)[:, np.newaxis] * np.std(lfp) * len(lfp))
    lags = signal.correlation_lags(lfp.size, lfp.size, mode="same")

    center = len(lfp) // 2
    order = np.argsort(np.mean(corr_norm[:, center-200:center+200], axis=1)) if sorting == "max" else sidx
    clim = np.percentile(corr_norm.flatten(), [5, 95])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(corr_norm[order, :], aspect='auto', cmap='bwr',
                   extent=[lags[0] * bin_size, lags[-1] * bin_size, corr_norm.shape[0], 0], vmin=clim[0], vmax=clim[1])
    if sorting != "max":
        ax.set_yticks(middle)
        ax.set_yticklabels(areas, rotation=45, ha='right')
    ax.plot([0, 0], [0, corr_norm.shape[0]], '--k')
    ax.set_xlabel('Lag relative to LFP (s)')
    ax.set_ylabel('Unit')
    ax.set_xlim([-1, 1])
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04).set_label('Correlation')
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_mean_attr_clusters(mean_attr_clusters, lfp_obj, bands, show_plot=True, save_fig=False,
                            output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """|Mean attribution| per cluster (rows 1-5; row 0 is unused) and channel, one panel per model target."""
    num_channels = len(lfp_obj.channels)
    fig, ax = _band_grid(len(bands)+1, figsize=(10, 20))
    for bandi in range(len(bands)+1):
        im = ax.flat[bandi].imshow(np.abs(mean_attr_clusters[:, bandi, :]), aspect='auto', cmap='bwr')
        ax.flat[bandi].set_xticks(range(num_channels))
        ax.flat[bandi].set_xticklabels(lfp_obj.channels['dorsal_ventral_ccf_coordinate'].astype(int), rotation=45, ha='right')
        ax.flat[bandi].set_xlabel('dorsal ventral ccf coordinate')
        ax.flat[bandi].set_ylim([.5, 5.5])
        ax.flat[bandi].set_title(_band_title(bands, bandi))
        ax.flat[bandi].set_yticks(range(1, 6))
        ax.flat[bandi].set_yticklabels(range(1, 6))
        ax.flat[bandi].set_ylabel('Cluster ID')
        fig.colorbar(im, ax=ax.flat[bandi])
    plt.tight_layout()
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)


def plot_peri_stim_attr(attr, lfp, session_obj, timestamps, time_win, bin_size=config.BIN_SIZE, show_plot=True,
                        save_fig=False, output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """
    Stimulus-triggered attribution per area (z-scored per unit, mean +- SEM across units) with the
    stimulus-triggered LFP overlaid, for every presentation that fits within timestamps.

    Returns (attr_snippets, lfp_snippets, t): presentations x time (x units).
    """
    t = np.arange(time_win[0], time_win[1], bin_size)

    stim_st = session_obj.stimulus_presentation.start_time
    stim_st = stim_st[((stim_st - np.abs(time_win[0])) > timestamps[0]) &
                      ((stim_st + np.abs(time_win[1])) < timestamps[-1])].values

    areas = sorted(session_obj.units['structure_acronym'].unique())
    attr_snippets = np.zeros((len(stim_st), len(t), attr.shape[1]))
    lfp_snippets = np.zeros((len(stim_st), len(t)))
    for i, st in enumerate(stim_st):
        start = np.argmin(np.abs(timestamps - (st - np.abs(time_win[0]))))
        attr_snippets[i] = attr[start:start+len(t), :]
        lfp_snippets[i] = lfp[start:start+len(t)]

    attr_snippets_norm = zscore(np.nanmean(attr_snippets, axis=0), axis=0)
    keep = ~np.any(np.isnan(attr_snippets_norm), axis=0)    # drops units with constant attribution
    mean_lfp = np.mean(lfp_snippets, axis=0)

    fig, ax = _band_grid(len(areas), figsize=(16, 16))
    for areai, area in enumerate(areas):
        tmp = attr_snippets_norm[:, (session_obj.units['structure_acronym'].values == area) & keep]
        mean, sem = np.mean(tmp, axis=1), np.std(tmp, axis=1) / np.sqrt(tmp.shape[1])
        ax.flat[areai].plot(t, mean, 'r', label='mean')
        ax.flat[areai].fill_between(t, mean - sem, mean + sem, color='r', alpha=0.5)
        ax2 = ax.flat[areai].twinx()
        ax2.plot(t, mean_lfp, 'k')
        ax.flat[areai].set_ylabel('Mean attribution')
        ax2.set_ylabel('Mean LFP')
        ax.flat[areai].set_xlim([time_win[0], time_win[1]])
        corr = np.corrcoef(mean_lfp, mean)[0, 1]
        ax.flat[areai].set_title(f'{area} Corr: {np.round(corr, 2)}')
    plt.tight_layout()
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)

    return attr_snippets, lfp_snippets, t


def plot_unit_attr_fr_corr(unit_attr_fr_corr, session_obj, bands, show_plot=True, save_fig=False,
                           output_dir=None, session_id=None, fig_name='default_name.pdf'):
    """Per-unit correlation between attribution and firing rate (units x channels x targets), one panel per channel."""
    num_channels = unit_attr_fr_corr.shape[1]
    num_bands = unit_attr_fr_corr.shape[2]
    sidx = _area_order(session_obj.units['structure_acronym'].values)[0]

    fig, ax = plt.subplots(int(np.ceil(num_channels/2)), 2, figsize=(10, 40))
    for chani in range(num_channels):
        ax.flat[chani].imshow(unit_attr_fr_corr[sidx, chani, :], aspect='auto', cmap='bwr',
                              extent=[0, num_bands, unit_attr_fr_corr.shape[0], 0], vmin=-.1, vmax=.1,
                              interpolation='none')
        ax.flat[chani].set_xticks(range(num_bands))
        ax.flat[chani].set_xticklabels(['Broadband'] + [str(b) for b in bands], rotation=45, ha='center')
        ax.flat[chani].set_title('Channel ' + str(chani))
    plt.tight_layout()
    _finish(fig, show_plot, save_fig, output_dir, session_id, fig_name)
