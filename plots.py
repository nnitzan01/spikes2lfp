import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    session_path = f"{output_dir}/plots/{session_id}"
    file_path = f"{session_path}/{plot_name}"
    os.makedirs(session_path, exist_ok=True)
    fig.savefig(file_path)

def plot_r2(scoresTest, channels, bands, data_type='active', show_plot=True, 
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
    cax.set_clim(-.4, .7);
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)

def plot_lfp_prediction(y, yHat, chani, bands, start_time = None, end_time = None, fs=250, show_plot=True,
                        save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    """
    Plot the LFP prediction for the given channel and bands.

    input:
    y: torch.Tensor, target data, assume shape is (#trial, seqlength, #units)
                                will be reshaped to (#trial*seqlength, #units)
    yHat: torch.Tensor, predicted data, assume shape is (#trial, seqlength, #units)
                                will be reshaped to (#trial*seqlength, #units)
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
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(16, 16))
    # if len(X.shape) == 3:
    #     X = torch.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    for bandi in range(len(bands)+1):
        # yHat = models[chani, bandi].evaluate(X)
        if start_time != None and end_time != None:
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            # X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]
            yHat_test = yHat[start_idx:end_idx, chani, bandi]
        else:
            start_time = 0
            end_time = y.shape[0]/fs
            start_idx = 0
            end_idx = y.shape[0]
        ax.flat[bandi].plot(np.linspace(start_time, end_time, int(y_test.shape[0])), y_test.cpu().detach().numpy()[0:int(end_idx-start_idx), chani, bandi],'k',label='LFP')
        ax.flat[bandi].plot(np.linspace(start_time, end_time, int(y_test.shape[0])), yHat_test[0:int(end_idx-start_idx)],'r',label='LSTM')
        if bandi == 0:
            ax.flat[bandi].set_title('Channel ' + str(chani) + ' Broadband')
        else:
            ax.flat[bandi].set_title('Channel ' + str(chani) + ' Band: ' + str(bands[bandi-1]) + ' Hz')
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

def plot_abs_error_change(y, yHat, session_obj, timestamps, chani, bands,
                   snippet_length=0.75, show_plot=True, save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    abs_error = np.zeros((y.shape[0], len(bands)+1))
    for bandi in range(len(bands)+1):
        abs_error[:, bandi] = np.abs(yHat[:, bandi] - y[:, chani, bandi])

    start_times = session_obj.active.start_time.values
    change = np.array(session_obj.active.is_change.values, dtype=bool)

    bin_size = 0.004
    nbins = int(snippet_length/bin_size)
    t = np.linspace(-0.25, 0.5, nbins)
    
    error_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    lfp_snippets = np.zeros((len(start_times), nbins, len(bands)+1))

    for i in range(len(start_times)):
        start = np.argmin(np.abs(timestamps - (start_times[i]-.25)))
        error_snippets[i,:,:] = abs_error[start:start+nbins,:]
        lfp_snippets[i,:,:] = y[start:start+nbins,chani,:]
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(30, 30))

    for bandi in range(len(bands)+1):
        ax.flat[bandi].imshow(error_snippets[:, :,bandi], aspect='auto', cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2, alpha=0.9)
        
        avr_condition = np.mean(error_snippets[change, :,bandi], axis=0)
        avr_nochange = np.mean(error_snippets[~change, :,bandi], axis=0)
        
        ax.flat[bandi].plot(t, len(start_times)//2 + avr_condition * len(start_times)//5,   'm', label='change', linewidth=3)
        ax.flat[bandi].plot(t, len(start_times)//2 + avr_nochange * len(start_times)//5, 'k', label='no change', linewidth=3)
        ax.flat[bandi].set_xlabel('Time from change (s)')
        if bandi==0:
            ax.flat[bandi].set_ylabel('Trial')
            ax.flat[bandi].set_title('broaband')
            ax.flat[bandi].legend(loc = 'upper left')
        else:
            ax.flat[bandi].set_title('Band: ' + str(bands[bandi-1]) + ' Hz')

    # add colorbar to the last subplot
    ax.flat[-1].imshow(np.zeros((1,1)), cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2)
    # turn off the axis
    ax.flat[-1].axis('off')
    # add colorbar
    cbar = fig.colorbar(ax.flat[-1].images[0], ax=ax.flat[-1])
    # fig.delaxes(ax.flat[-1])
    # plt.tight_layout()
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

    # add colorbar to the last subplot
    ax.flat[-1].imshow(np.zeros((1,1)), cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2)
    # turn off the axis
    ax.flat[-1].axis('off')
    # add colorbar
    cbar = fig.colorbar(ax.flat[-1].images[0], ax=ax.flat[-1])
    # fig.delaxes(ax.flat[-1])
    # plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)

def plot_abs_error(lfp, yHat, session_obj, timestamps, chani, bands, stim_type, stim_condition, 
                   snippet_length=0.75, show_plot=True, save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    abs_error = np.zeros((lfp.shape[0], len(bands)+1))
    for bandi in range(len(bands)+1):
        abs_error[:, bandi] = np.abs(yHat[:, chani, bandi] - lfp[:, chani, bandi])

    if stim_type == 'active':
        start_times = session_obj.active.start_time.values
    elif stim_type == 'spontaneous':
        start_times = session_obj.spontaneous.start_time.values
    elif stim_type == 'passive':
        start_times = session_obj.passive.start_time.values
    else: # default value
        start_times = session_obj.active.start_time.values

    if stim_condition == 'change':
        condition = np.array(session_obj.active.is_change.values, dtype=bool)
    elif stim_condition == 'omission':
        condition = np.array(session_obj.active.omitted.values, dtype=bool)
        start_times = start_times[condition]
    else:
        condition = np.ones(len(start_times), dtype=bool)

    fs = 250
    nbins = int(snippet_length*fs)
    t = np.linspace(-0.25, 0.5, nbins)
    error_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    lfp_snippets = np.zeros((len(start_times), nbins, len(bands)+1))
    print(error_snippets.shape)
    print(abs_error.shape)
    for i in range(len(start_times)):
        start = np.argmin(np.abs(timestamps - (start_times[i]-.25)))
        error_snippets[i,:,:] = abs_error[start:start+nbins,:]
        lfp_snippets[i,:,:] = lfp[start:start+nbins,chani,:]
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(30, 30))

    for bandi in range(len(bands)+1):
        avr_condition = np.mean(error_snippets[:, :,bandi], axis=0)
        ax.flat[bandi].plot(t, len(start_times)//2 + avr_condition * len(start_times)//5,   'm', label='change', linewidth=3)
        if stim_condition == 'change':
            ax.flat[bandi].imshow(error_snippets[:, :,bandi], aspect='auto', cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2, alpha=0.9)
            avr_nochange = np.mean(error_snippets[~condition, :,bandi], axis=0)
            ax.flat[bandi].plot(t, len(start_times)//2 + avr_nochange * len(start_times)//5, 'k', label='no change', linewidth=3)
        elif stim_condition == 'omission':
            ax.flat[bandi].imshow(error_snippets[:, :,bandi], aspect='auto', cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2)
            avr_omission_data = np.mean(lfp_snippets[:, :, bandi], axis=0)
            ax.flat[bandi].plot(t, len(start_times)//2 + avr_omission_data * len(start_times)//5, 'k', label='omission data', linewidth=3)
        ax.flat[bandi].set_xlabel('Time from change (s)')
        if bandi==0:
            ax.flat[bandi].set_ylabel('Trial')
            ax.flat[bandi].set_title('broaband')
            ax.flat[bandi].legend(loc = 'upper left')
        else:
            ax.flat[bandi].set_title('Band: ' + str(bands[bandi-1]) + ' Hz')

    # add colorbar to the last subplot
    ax.flat[-1].imshow(np.zeros((1,1)), cmap='bwr', extent=[-0.25, 0.5, 0, len(start_times)],vmin=0, vmax=2)
    # turn off the axis
    ax.flat[-1].axis('off')
    # add colorbar
    cbar = fig.colorbar(ax.flat[-1].images[0], ax=ax.flat[-1])
    # fig.delaxes(ax.flat[-1])
    # plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    if save_fig:
        save_plot(fig, output_dir, session_id, fig_name)