import os
import numpy as np
import torch
from pathlib import Path
from scipy.signal import welch
from scipy.stats import zscore
from scipy import signal 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def save_plot(fig, output_dir, session_id, plot_name = 'default_name.png'):
    """
    Save the plot to the output directory.

    input:
    fig: matplotlib figure object
    output_dir: str, parent folder to save the plot

    output:
    None

    file structure:
    output_dir/plot.png
    """
    session_path = Path(output_dir / "plots" / session_id)
    file_path = Path(session_path / plot_name)
    os.makedirs(session_path, exist_ok=True)
    fig.savefig(file_path)

def plot_r2(scoresTest, channels, bands, clim = [-.4, .7],  data_type='active', show_plot=True, 
            save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    """
    Make and/or save a plot of the R2 scores for the models.

    input:
    scoreTest: np.array, shape is (#channels, #bands+1), contains the R2 scores
    channels: pandas DataFrame, channel information
    bands: list, frequency bands
    data_type: str, 'active', 'spontaneous', etc.
    show_plot: bool, display the plot
    save_fig: bool, save the plot
    output_dir: str, parent folder to save the plot
    session_id: int, session_id
    fig_name: str, name of the plot
    """
    num_channels = len(channels)
    assert scoresTest.shape == (num_channels, len(bands)+1)
    fig,ax = plt.subplots(1,1,figsize=(7,7))
    cax = ax.imshow(scoresTest, aspect='equal', cmap='viridis')
    ax.set_xlabel('Band')
    ax.set_ylabel('Channel')
    ax.set_title('Model score for all channels and bands on ' + data_type + ' data')
    ax.set_yticks(range(num_channels))
    ax.set_yticklabels(channels['dorsal_ventral_ccf_coordinate'], rotation=45)
    ax.set_xticks(range(len(bands)+1))
    ax.set_xticklabels(['Broadband'] + [str(bands[i]) for i in range(len(bands))], rotation=45, ha='right')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    cbar = fig.colorbar(cax)
    cbar.set_label('R2')
    cax.set_clim(clim[0], clim[1])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)

def plot_lfp_prediction(y, yHat, chan2use, bands, start_time = None, end_time = None, fs=250, show_plot=True,
                        save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    """
    Plot the LFP prediction for the given channel and bands.

    input:
    y: torch.Tensor, target data
    yHat: torch.Tensor, predicted data
    chani: int, channel index
    bands: list, frequency bands
    start_time: float, in seconds, start time for the plot, assume 0 is the start of X
    end_time: float, in seconds, end time for the plot, assume end of X is the end
    fs: int, sampling frequency
    show_plot: bool, display the plot
    save_fig: bool, save the plot
    output_dir: str, parent folder to save the plot
    session_id: int, session_id
    fig_name: str, name of the plot
    """
    # if not already on the cpu, move it there
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
        
    if len(yHat.shape) == 2:
        chan2use = 0
        yHat = yHat[:, np.newaxis]

       
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(16, 16))
    for bandi in range(len(bands)+1):
        if start_time != None and end_time != None:
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            y_test = y[start_idx:end_idx]
            yHat_test = yHat[start_idx:end_idx]
        else:
            start_time = 0
            end_time = y.shape[0]/fs
            start_idx = 0
            end_idx = y.shape[0]
        ax.flat[bandi].plot(np.linspace(start_time, end_time, int(y_test.shape[0])), y_test[0:int(end_idx-start_idx), chan2use, bandi],'k',label='LFP')
        ax.flat[bandi].plot(np.linspace(start_time, end_time, int(y_test.shape[0])), yHat_test[0:int(end_idx-start_idx), chan2use, bandi],'r',label='LSTM')
        if bandi == 0:
            ax.flat[bandi].set_title('Channel ' + str(chan2use) + ' Broadband')
        else:
            ax.flat[bandi].set_title('Channel ' + str(chan2use) + ' Band: ' + str(bands[bandi-1]) + ' Hz')
    fig.delaxes(ax.flat[-1])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)

def plot_all_channel_loss(channels, bands, bandi, losses, show_plot=True, 
                        save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    """
    Plot the loss for all channels for a given band.

    input:
    channels: pandas DataFrame, channel information
    bands: list, frequency bands
    bandi: int, selected band index
    losses: np.array, shape is (#channels, #epochs, #bands)
    show_plot: bool, display the plot
    save_fig: bool, save the plot
    output_dir: str, parent folder to save the plot
    session_id: int, session_id
    fig_name: str, name of the plot
    """
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    norm = mcolors.Normalize(vmin=channels['dorsal_ventral_ccf_coordinate'].min(), 
                             vmax=channels['dorsal_ventral_ccf_coordinate'].max())
    cmap = cm.viridis
    for chani in range(len(channels)):
        color = cmap(norm(channels['dorsal_ventral_ccf_coordinate'].iloc[chani]))
        ax.plot(losses[chani, :, bandi], 'o-', alpha=0.5, color=color)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    if bandi==0:
        ax.set_title('Model loss for broadband model')
    else:
        ax.set_title('Model loss for ' + str(bands[bandi-1]) + ' Hz model')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Electrode Depth')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)

def plot_abs_error_change(y, yHat, session_obj, timestamps, chan2use, bands,
                   snippet_length=0.75, show_plot=True, save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    abs_error = np.zeros((y.shape[0], len(bands)+1))
    for bandi in range(len(bands)+1):
        abs_error[:, bandi] = np.abs(yHat[:,chan2use, bandi] - y[:, chan2use, bandi])
    start_times = session_obj.active.start_time.values
    start_times = start_times[start_times < timestamps[-1]-snippet_length]
    change = np.array(session_obj.active.is_change.values, dtype=bool)
    change = change[:len(start_times)]
    
    bin_size = 0.004
    nbins = int(snippet_length/bin_size)
    t = np.linspace(-0.25, 0.5, nbins)
    error_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    lfp_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    for i in range(len(start_times)):
        start = np.argmin(np.abs(timestamps - (start_times[i]-.25)))
        error_snippets[i,:,:] = abs_error[start:start+nbins,:]
        lfp_snippets[i,:,:] = y[start:start+nbins,chan2use,:]
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(30, 30))
    for bandi in range(len(bands)+1):
        ax.flat[bandi].imshow(error_snippets[:, :,bandi], aspect='auto', cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2, alpha=0.9)
        avr_condition = np.mean(error_snippets[change, :,bandi], axis=0)
        avr_nochange = np.mean(error_snippets[~change, :,bandi], axis=0)
        ax2 = ax.flat[bandi].twinx()
        ax2.plot(t, avr_condition ,   'm', label='change', linewidth=3)
        ax2.plot(t, avr_nochange, 'k', label='no change', linewidth=3)
        # ax.flat[bandi].plot(t, len(start_times)//2 + avr_condition * len(start_times)//5,   'm', label='change', linewidth=3)
        # ax.flat[bandi].plot(t, len(start_times)//2 + avr_nochange * len(start_times)//5, 'k', label='no change', linewidth=3)
        ax.flat[bandi].set_xlabel('Time from change (s)')
        if bandi==0:
            ax.flat[bandi].set_ylabel('Trial')
            ax.flat[bandi].set_title('broaband')
            ax.flat[bandi].legend(loc = 'upper left')
        else:
            ax.flat[bandi].set_title('Band: ' + str(bands[bandi-1]) + ' Hz')
    ax.flat[-1].imshow(np.zeros((1,1)), cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2)
    ax.flat[-1].axis('off')
    cbar = fig.colorbar(ax.flat[-1].images[0], ax=ax.flat[-1])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)

def plot_abs_error_omission(lfp, yHat, session_obj, timestamps, chani, bands,
                   snippet_length=0.75, show_plot=True, save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    abs_error = np.zeros((lfp.shape[0], len(bands)+1))
    for bandi in range(len(bands)+1):
        abs_error[:, bandi] = np.abs(yHat[:, bandi] - lfp[:, chani, bandi])
    omission = np.array(session_obj.active.omitted.values, dtype=bool)
    start_times = session_obj.active.start_time.values[omission]
    fs = 250
    nbins = int(snippet_length*fs)
    t = np.linspace(-0.25, 0.5, nbins)
    error_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    lfp_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    for i in range(len(start_times)):
        start = np.argmin(np.abs(timestamps - (start_times[i]-.25)))
        error_snippets[i,:,:] = abs_error[start:start+nbins,:]
        lfp_snippets[i,:,:] = lfp[start:start+nbins,chani,:]
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(30, 30))
    for bandi in range(len(bands)+1):
        ax.flat[bandi].imshow(error_snippets[:, :,bandi], aspect='auto', cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2)
        avr_omission_data = np.mean(lfp_snippets[:, :, bandi], axis=0)
        ax.flat[bandi].plot(t, len(start_times)//2 + avr_omission_data * len(start_times)//5, 'k', label='omission data', linewidth=3)
        avr_condition = np.mean(error_snippets[:, :,bandi], axis=0)
        ax.flat[bandi].plot(t, len(start_times)//2 + avr_condition * len(start_times)//5,   'w', label='change', linewidth=3)
        ax.flat[bandi].set_xlabel('Time from change (s)')
        if bandi==0:
            ax.flat[bandi].set_ylabel('Trial')
            ax.flat[bandi].set_title('broaband')
            ax.flat[bandi].legend(loc = 'upper left')
        else:
            ax.flat[bandi].set_title('Band: ' + str(bands[bandi-1]) + ' Hz')
    ax.flat[-1].imshow(np.zeros((1,1)), cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2)
    ax.flat[-1].axis('off')
    cbar = fig.colorbar(ax.flat[-1].images[0], ax=ax.flat[-1])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        
def plot_psd(y, yHat, chani, show_plot=True, save_fig=False, 
             output_dir=None, session_id=None, fig_name='default_name.png'):

    f, Pxx = welch(y[:,chani, 0], 250, nperseg=64)
    f, Pxx_hat = welch(yHat[:,chani, 0], 250, nperseg=64)

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(f, np.log10(Pxx), 'r', label='LFP')
    ax.plot(f, np.log10(Pxx_hat), 'g', label='model')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log10(PSD)')
    ax.set_title('Power spectral density of LFP and model prediction')
    ax.set_xlim([0, 120])
    ax.legend()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        
        
def plot_attr_corrmat(attr,session_obj, show_plot=True, save_fig=False, 
                      output_dir=None, session_id=None, fig_name='default_name.png'):
    
    locs = session_obj.units['structure_acronym']
    sidx = np.argsort(locs.values)
    sorted_locs = locs.values[sidx]

    # for each area, find the first and last unit
    areas = np.unique(sorted_locs)
    first = []
    last  = []
    for area in areas:
        ind = sorted_locs == area
        first.append(np.where(ind)[0][0])
        last.append(np.where(ind)[0][-1])
    middle = (np.array(first) + (np.array(last) - np.array(first)) / 2).astype(int)
    
    attr = attr[:, sidx]
    corrmat = np.corrcoef(attr.T)
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(corrmat, vmin=-.1, vmax=.1, aspect='auto', cmap='bwr')
    ax.set_xticks(middle)
    ax.set_xticklabels(areas, rotation=45, ha='right')
    ax.set_yticks(middle)
    ax.set_yticklabels(areas, rotation=45, ha='right')
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.026, pad=0.04)
    cbar.set_label('Correlation')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        
def plot_attr_density_plots(mean_attribution, session_obj, bands, area, show_plot=True, save_fig=False, 
                            output_dir=None, session_id=None, fig_name='default_name.png'):
    
    exp   = np.ceil(np.log10(np.abs(np.median(mean_attribution))))

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    
    for bandi in range(len(bands)+1):
        if bandi == 0:
            plt.hist(np.abs(mean_attribution[:,bandi,session_obj.units['structure_acronym'].str.contains(area)].flatten()), 
            bins=50, range = (0, 10 ** exp), density=False, histtype = 'step', label='Broadband',
            weights=np.ones(len(mean_attribution[:,bandi,session_obj.units['structure_acronym'].str.contains(area)].flatten())) / len(mean_attribution[:,bandi,session_obj.units['structure_acronym'].str.contains(area)].flatten()))
        else:
            plt.hist(np.abs(mean_attribution[:,bandi,session_obj.units['structure_acronym'].str.contains(area)].flatten()), 
            bins=50, range = (0, 10 ** exp), density=False, histtype = 'step', label=f'{bands[bandi-1][0]}-{bands[bandi-1][1]} Hz',
            weights=np.ones(len(mean_attribution[:,bandi,session_obj.units['structure_acronym'].str.contains(area)].flatten())) / len(mean_attribution[:,bandi,session_obj.units['structure_acronym'].str.contains(area)].flatten()))

    ax.set_xlabel('Attribution')
    ax.set_ylabel('Probability')
    ax.set_xlim([- 10 ** (exp-2), 10 ** exp])
    # set the y-axis to be log scale
    # ax.set_yscale('log')
    ax.legend(loc='upper right', frameon=False)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        
        
def plot_mean_attr_areas(mean_attr_areas, lfp_obj, session_obj, bands, show_plot=True, save_fig=False, 
                         output_dir=None, session_id=None, fig_name='default_name.png'):
    
    num_channels = len(lfp_obj.channels)
    areas = sorted(session_obj.units['structure_acronym'].unique())
    
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2,figsize=(10,20))
    for bandi in range(len(bands)+1):
        exp   = np.ceil(np.log10(np.abs(np.median(mean_attr_areas[:,bandi,:].flatten()))))
        
        ax.flat[bandi].imshow(np.abs(mean_attr_areas[:,bandi,:]) , aspect='auto', cmap='bwr')
        # set the xticks to be the channel depths
        ax.flat[bandi].set_xticks(range(num_channels))
        ax.flat[bandi].set_xticklabels(lfp_obj.channels['dorsal_ventral_ccf_coordinate'].astype(int), 
                                    rotation=45, ha='right')
        ax.flat[bandi].set_xlabel('dorsal ventral ccf coordinate')

        # set the yticks to be the area names
        ax.flat[bandi].set_yticks(range(len(areas)))
        ax.flat[bandi].set_yticklabels(areas)
        ax.flat[bandi].set_ylabel('Area')
        if bandi == 0:
            ax.flat[bandi].set_title('Broadband')
        else:
            ax.flat[bandi].set_title(str(bands[bandi-1]) + ' Hz')
            
        cbar = fig.colorbar(ax.flat[bandi].imshow(np.abs(mean_attr_areas[:,bandi,:]), 
                                                aspect='auto', cmap='bwr'), 
                            ax=ax.flat[bandi])

    # tight layout
    plt.tight_layout()
    fig.delaxes(ax.flat[-1])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        

def plot_mean_attr_pyr_int(mean_attribution, session_obj, area, df, bands, chan2use, show_plot=True, save_fig=False, 
                         output_dir=None, session_id=None, fig_name='default_name.png'):
    
    area_idx = session_obj.units.index[session_obj.units['structure_acronym'].str.contains(area)]
    pyr  = np.intersect1d(df.unit_id[df.pyr == 1], area_idx)
    intn = np.intersect1d(df.unit_id[df.pyr == 0], area_idx)

    attr_pyr = mean_attribution[:, :,session_obj.units.index.isin(pyr)]
    attr_int = mean_attribution[:, :,session_obj.units.index.isin(intn)]

    # for each band, plot boxplots of the mean attribution for pyramidal cells and interneurons
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(5,8))

    for bandi in range(len(bands)+1):
        ax.flat[bandi].boxplot([np.abs(attr_pyr[chan2use, bandi,:]), np.abs(attr_int[chan2use, bandi,:])], showfliers=False)
        if bandi == 0:
            ax.flat[bandi].set_title('Broadband')
        else:
            ax.flat[bandi].set_title(str(bands[bandi-1]) + ' Hz')
        ax.flat[bandi].set_xticklabels(['pyr', 'int'])
        ax.flat[bandi].set_ylabel('Mean attribution')
        
    # tight layout
    plt.tight_layout()
    fig.delaxes(ax.flat[-1])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        
        
def plot_attr_matrix(attr, timestamps, session_obj, time_win, plot_stim = False, bin_size = 0.004, show_plot=True, save_fig=False):
    
    exp   = np.ceil(np.log10(np.abs(np.median(attr[attr!=0]))))

    stim_st = session_obj.stimulus_presentation.start_time
    stim_st = stim_st[(stim_st > time_win[0]) & (stim_st < time_win[1])]
    stim_st = stim_st.values
    
    locs = session_obj.units['structure_acronym']
    sidx = np.argsort(locs.values)
    sorted_locs = locs.values[sidx]

    # for each area, find the first and last unit
    areas = np.unique(sorted_locs)
    first = []
    last  = []
    for area in areas:
        ind = sorted_locs == area
        first.append(np.where(ind)[0][0])
        last.append(np.where(ind)[0][-1])
    middle = (np.array(first) + (np.array(last) - np.array(first)) / 2).astype(int)
    
    st = int((time_win[0] - timestamps[0]) / bin_size)
    en = int((time_win[1] - timestamps[0]) / bin_size)
    
    # visualize the attribution we just loaded for the broadband model
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.imshow(attr[st:en,sidx].T, aspect='auto', cmap='bwr', vmin=-0.03, vmax=0.03,
            extent = [time_win[0], time_win[1], len(session_obj.units), 0])
    if plot_stim:
        for st in stim_st:
            ax.axvline(st,     color='k', linestyle='--')
            ax.axvline(st+.25, color='k', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit') 
    ax.set_xlim([time_win[0], time_win[1]])
    ax.set_yticks(middle)
    ax.set_yticklabels(areas, rotation=45, ha='right')
    plt.show()
        
    
def plot_attr_lfp_corr(attr, lfp, session_obj, sorting = "max", bin_size = 0.004,
                        show_plot=True, save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    
    locs = session_obj.units['structure_acronym']
    sidx = np.argsort(locs.values)
    sorted_locs = locs.values[sidx]

    # for each area, find the first and last unit
    areas = np.unique(sorted_locs)
    first = []
    last  = []
    for area in areas:
        ind = sorted_locs == area
        first.append(np.where(ind)[0][0])
        last.append(np.where(ind)[0][-1])
    middle = (np.array(first) + (np.array(last) - np.array(first)) / 2).astype(int)
    
    if len(lfp.shape) == 1:
        lfp = lfp[:, np.newaxis]
        
    cross_corr = np.zeros((attr.shape[1], lfp.shape[0]))
    for i in range(attr.shape[1]):
        cross_corr[i,:] = signal.correlate(lfp.flatten(), attr[:,i], mode='same')
        
    corr_norm = cross_corr / (np.std(attr, axis=0)[:, np.newaxis] * np.std(lfp) * len(lfp))
    lags = signal.correlation_lags(lfp.size, lfp.size, mode="same")
    
    midx = np.argsort(np.mean(corr_norm[:,int(lfp.shape[0]/2)-200:int(lfp.shape[0]/2)+200], axis=1))
    
    # for clim use the 5th and 95th percentile
    clim = np.percentile(corr_norm.flatten(), [5, 95])
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    if sorting == "max":
        ax.imshow(corr_norm[midx,:], aspect='auto', cmap='bwr',
          extent=[lags[0] * bin_size, lags[-1] * bin_size, corr_norm.shape[0],0], vmin=clim[0], vmax=clim[1])
    else:
       ax.imshow(corr_norm[sidx,:], aspect='auto', cmap='bwr',
          extent=[lags[0] * bin_size, lags[-1] * bin_size, corr_norm.shape[0],0], vmin=clim[0], vmax=clim[1])
       ax.set_yticks(middle)
       ax.set_yticklabels(areas, rotation=45, ha='right')
        
    ax.plot([0, 0], [0, corr_norm.shape[0]], '--k')
    ax.set_xlabel('Lag relative to LFP (s)')
    ax.set_ylabel('Unit')
    ax.set_xlim([-1, 1])
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_ylabel('Unit')

    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.026, pad=0.04)
    cbar.set_label('Correlation')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        

def plot_mean_attr_clusters(mean_attr_clusters, lfp_obj, bands, show_plot=True, save_fig=False, 
                         output_dir=None, session_id=None, fig_name='default_name.png'):
    
    num_channels = len(lfp_obj.channels) 
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2,figsize=(10,20)) 
    
    for bandi in range(len(bands)+1):
        ax.flat[bandi].imshow( np.abs(mean_attr_clusters[:,bandi,:]) , aspect='auto',cmap='bwr')
        # set the xticks to be the channel depths
        ax.flat[bandi].set_xticks(range(num_channels))
        ax.flat[bandi].set_xticklabels(lfp_obj.channels['dorsal_ventral_ccf_coordinate'].astype(int) , rotation=45, ha='right')
        ax.flat[bandi].set_xlabel('dorsal ventral ccf coordinate')
        ax.flat[bandi].set_ylim([.5,5.5])
        if bandi == 0:
            ax.flat[bandi].set_title('Broadband')
        else:
            ax.flat[bandi].set_title(str(bands[bandi-1]) + ' Hz')
        # set the yticks to be the cluster ids
        ax.flat[bandi].set_yticks(range(1,6,1))
        ax.flat[bandi].set_yticklabels(range(1,6,1))
        ax.flat[bandi].set_ylabel('Cluster ID')
        cbar = fig.colorbar(ax.flat[bandi].imshow(np.abs(mean_attr_clusters[:,bandi,:]), aspect='auto', cmap='bwr'), ax=ax.flat[bandi])
        
    # tight layout
    plt.tight_layout()
    fig.delaxes(ax.flat[-1])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
        
        
def plot_peri_stim_attr(attr, lfp, session_obj, timestamps, time_win, bin_size = 0.004, show_plot=True, save_fig=False, 
                         output_dir=None, session_id=None, fig_name='default_name.png'):
    
    t = np.arange(time_win[0], time_win[1], bin_size)

    stim_st = session_obj.stimulus_presentation.start_time
    stim_st = stim_st[((stim_st - np.abs(time_win[0])) > timestamps[0]) & ((stim_st + np.abs(time_win[1])) < timestamps[-1])]
    stim_st = stim_st.values
    
    areas = sorted(session_obj.units['structure_acronym'].unique())
    
    attr_snippets = np.zeros((len(stim_st), len(t), attr.shape[1])) # trials x time x neurons
    lfp_snippets = np.zeros((len(stim_st), len(t)))
    
    for i in range(len(stim_st)):
        start = np.argmin(np.abs(timestamps - (stim_st[i]- np.abs(time_win[0])  )))
        attr_snippets[i,:,:] = attr[start:start+len(t),:]
        lfp_snippets[i,:]    = lfp[start:start+len(t)]
    
    attr_snippets_avr  = np.nanmean(attr_snippets, axis=0)
    attr_snippets_norm = zscore(attr_snippets_avr, axis=0)
    keep = ~np.any(np.isnan(attr_snippets_norm),axis=0)
    
    mean_lfp = np.mean(lfp_snippets, axis=0)
    
    fig, ax = plt.subplots(int(np.ceil(len(areas)/2)) ,2,figsize=(16,16))
    for areai in range(len(areas)):
        # get mean across trials
        inarea = session_obj.units['structure_acronym'].values == areas[areai]
        num_neurons = np.sum(inarea)
        tmp = attr_snippets_norm[:, inarea & keep]
        # num_neurons = np.sum(session_obj.units['structure_acronym'] == areas[areai])
        # plot mean and sem
        ax.flat[areai].plot(t, np.mean(tmp, axis=1), 'r', label='mean')
        ax.flat[areai].fill_between(t, np.mean(tmp, axis=1) - np.std(tmp, axis=1)/np.sqrt(num_neurons), 
                                 np.mean(tmp, axis=1) + np.std(tmp, axis=1)/np.sqrt(num_neurons), color='r', alpha=0.5)
        ax2 = ax.flat[areai].twinx()
        ax2.plot(t, mean_lfp, 'k')
        ax.flat[areai].set_ylabel('Mean attribution')
        ax2.set_ylabel('Mean LFP')
        ax.flat[areai].set_xlim([-0.25, 0.5])
        corr = np.corrcoef(mean_lfp, np.mean(tmp, axis=1))[0,1]
        ax.flat[areai].set_title(areas[areai] + ' Corr: ' + str(np.round(corr,2)))
    
    if len(areas) % 2 != 0:
       fig.delaxes(ax.flat[-1]) 
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)
    
    return attr_snippets, lfp_snippets, t
    
    
    
def plot_unit_attr_fr_corr(unit_attr_fr_corr, session_obj, bands, show_plot=True, save_fig=False, 
                         output_dir=None, session_id=None, fig_name='default_name.png'):
    
    num_channels = unit_attr_fr_corr.shape[1]
    num_bands    = unit_attr_fr_corr.shape[2]
    
    locs = session_obj.units['structure_acronym']
    sidx = np.argsort(locs.values)
    sorted_locs = locs.values[sidx]

    # for each area, find the first and last unit
    areas = np.unique(sorted_locs)
    first = []
    last  = []
    for area in areas:
        ind = sorted_locs == area
        first.append(np.where(ind)[0][0])
        last.append(np.where(ind)[0][-1])
    middle = (np.array(first) + (np.array(last) - np.array(first)) / 2).astype(int)
    
    fig, ax = plt.subplots(int(np.ceil(num_channels/2)) ,2,figsize=(10,40))
    
    for chani in range(num_channels):
        ax.flat[chani].imshow(unit_attr_fr_corr[sidx,chani,:], aspect='auto', cmap='bwr',
          extent=[0, num_bands, unit_attr_fr_corr.shape[0],0], vmin=-.1, vmax=.1, interpolation='none' )
        # ax.flat[chani].set_yticks(middle)
        # ax.flat[chani].set_yticklabels(areas, rotation=45, ha='right')
        ax.flat[chani].set_xticks(range(num_bands))
        ax.flat[chani].set_xticklabels(['Broadband'] + [str(bands[i]) for i in range(len(bands))], rotation=45, ha='center')
        ax.flat[chani].set_title('Channel ' + str(chani))
        
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)   
    
def visualize_filters(model, num_neurons):
    """Visualizes the filters of the first convolutional layer in the model,
    arranged in a square grid.

    Args:
        model: Trained PyTorch model with a convolutional layer.

    """
    # Set model to evaluation mode
    model.eval()

    # Access the convolutional layer
    conv_layer = model.conv1d

    # Extract the filters
    filters = conv_layer.weight.data

    # Get the number of filters, input channels, and kernel size
    out_channels, in_channels, kernel_size = filters.shape
    print (f"Channels: {out_channels} in: {in_channels}, kernel {kernel_size}")

    # Calculate the number of rows and columns for a square grid
    num_filters = out_channels
    num_cols = int(np.ceil(np.sqrt(num_filters)))
    num_rows = int(np.ceil(num_filters / num_cols))

    # Create the subplots for the grid
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows), layout='constrained') # set scaling factor manually to fit
    fig.suptitle("Heat Maps of Convolutional Layer Filters", fontsize=16)

    # Iterate and plot each heatmap. Ensure out_channels and in_channels are not too large.
    for i in range(num_filters):
        # Get the row and column index of the current filter
        row_idx = i // num_cols
        col_idx = i % num_cols

        # Get single filter
        currentFilter = filters[i,:,:]

        # convert to numpy
        currentFilter = currentFilter.cpu().numpy()

        # Normalize if it is exploding
        currentMax = np.max(np.abs(currentFilter))
        if (currentMax > 100):
            currentFilter = currentFilter/currentMax

        # Plot Heatmap
        if num_rows == 1:
           im = axs[col_idx].imshow(currentFilter, cmap='viridis', aspect = "auto")
           axs[col_idx].set_title(f"Filt {i}")

           # Remove all ticks and labels
           axs[col_idx].set_xticks([])
           axs[col_idx].set_yticks([])
        elif num_cols == 1:
            im = axs[row_idx].imshow(currentFilter, cmap='viridis', aspect = "auto")
            axs[row_idx].set_title(f"Filt {i}")

           # Remove all ticks and labels
            axs[row_idx].set_xticks([])
            axs[row_idx].set_yticks([])
        else:
           im = axs[row_idx, col_idx].imshow(currentFilter, cmap='viridis', aspect = "auto")

            # Set Heatmap title
           axs[row_idx, col_idx].set_title(f"Filt {i}")

            # Remove all ticks and labels
           axs[row_idx, col_idx].set_xticks([])
           axs[row_idx, col_idx].set_yticks([])

        # fig.colorbar(im, ax=axs[-1,0]) # always puts the colorbar in same location

    # If num_filters is not a perfect square, then the final plots will show the heat map.
    for i in range(num_filters,num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols

        if num_rows == 1:
            axs[col_idx].axis("off")
        elif num_cols == 1:
            axs[row_idx].axis("off")
        else:
            axs[row_idx,col_idx].axis("off")

    # Display plots
    plt.show()