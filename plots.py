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

def plot_r2(models, X, y, channels, bands, data_type='active', show_plot=True, 
            save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    """
    Make and/or save a plot of the R2 scores for the models.

    input:
    models: dict, LSTM models, keys are (channel, band) pairs
    X: torch.Tensor, input data
    y: torch.Tensor, target data
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
    scoresTest = np.zeros((num_channels, len(bands)+1))
    for chani in range(num_channels):
        for bandi in range(len(bands)+1):
            yHat_test  = models[chani, bandi].evaluate(X)
            r2_test  = r2_score(y[:, chani, bandi].cpu().detach().numpy(), yHat_test)
            scoresTest[chani, bandi] = r2_test
    fig,ax = plt.subplots(1,1,figsize=(5,5))
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

def plot_lfp_prediction(models, X, y, chani, bands, start_time = None, end_time = None, fs=250, show_plot=True,
                        save_fig=False, output_dir=None, session_id=None, fig_name='default_name.png'):
    """
    Plot the LFP prediction for the given channel and bands.

    input:
    models: dict, LSTM models, keys are (channel, band) pairs
    X: torch.Tensor, input data, assume shape is (#trial, seqlength, #units)
                                will be reshaped to (#trial*seqlength, #units)
    y: torch.Tensor, target data, ^^
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
    for bandi in range(len(bands)+1):
        yHat = models[chani, bandi].evaluate(X)
        if len(X.shape) == 3:
            X = torch.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
            yHat = torch.reshape(yHat, (yHat.shape[0]*yHat.shape[1], yHat.shape[2], yHat.shape[3]))
            y = torch.reshape(y, (y.shape[0]*y.shape[1], y.shape[2], y.shape[3]))
        if start_time != None and end_time != None:
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]
            yHat_test = yHat[start_idx:end_idx]
        else:
            start_time = 0
            end_time = X.shape[0]/fs
            start_idx = 0
            end_idx = X.shape[0]

        ax.flat[bandi].plot(np.linspace(start_time, end_time, int(X_test.shape[0])), y_test.cpu().detach().numpy()[0:int(end_idx-start_idx), chani, bandi],'k',label='LFP')
        ax.flat[bandi].plot(np.linspace(start_time, end_time, int(X_test.shape[0])), yHat_test[0:int(end_idx-start_idx)],'r',label='LSTM')
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

def temp2(models, session_obj, spikes, lfp, chani, bands, trial_type=None, ):
    abs_error = np.zeros((spikes.shape[0]*spikes.shape[1], len(bands)+1))
    for bandi in range(len(bands)+1):
        yHat = models[chani, bandi].evaluate(torch.tensor(spikes).float())
        abs_error[:,bandi] = np.abs(yHat - lfp[:, bandi, chani])
    
    stim_starts = session_obj.

    stim_st  = behavior.stimulus_presentations.start_time[behavior.stimulus_presentations.active].values
    change   = behavior.stimulus_presentations.is_change[behavior.stimulus_presentations.active] 

    nbins = int(0.75/bin_size)
    t = np.linspace(-0.25, 0.5, nbins)
    error_snippets = np.zeros((len(stim_st), nbins, len(bands)+1))

    for i in range(len(stim_st)):
        start = np.argmin(np.abs(spikes_obj.timestamps - (stim_st[i]-.25)))
        error_snippets[i,:,:] = abs_error[start:start+nbins,:]

    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(30, 30))

    for bandi in range(len(bands)+1):
        ax.flat[bandi].imshow(error_snippets[:,:,bandi], aspect='auto', cmap='bwr', extent=[-0.25, 0.5, 0, len(stim_st)],vmin=0, vmax=2)
        avr_change = np.mean(error_snippets[change == True,:,bandi], axis=0)
        avr_nochange = np.mean(error_snippets[change == False,:,bandi], axis=0)
        ax.flat[bandi].plot(t, 2000 + avr_change * 800,   'm', label='change', linewidth=3)
        ax.flat[bandi].plot(t, 2000 + avr_nochange * 800, 'k', label='no change', linewidth=3)
        ax.flat[bandi].set_xlabel('Time from change (s)')
        # ax.flat[bandi].set_aspect(1.0/ax.flat[bandi].get_data_ratio(), adjustable='box')
        if bandi==0:
            ax.flat[bandi].set_ylabel('Trial')
            ax.flat[bandi].set_title('broaband')
            ax.flat[bandi].legend(loc = 'upper left')
        else:
            ax.flat[bandi].set_title('Band: ' + str(bands[bandi-1]) + ' Hz')

    # add colorbar to the last subplot
    ax.flat[-1].imshow(np.zeros((1,1)), cmap='bwr', extent=[-0.25, 0.5, 0, len(stim_st)],vmin=0, vmax=2)
    # turn off the axis
    ax.flat[-1].axis('off')
    # add colorbar
    cbar = fig.colorbar(ax.flat[-1].images[0], ax=ax.flat[-1])
    # fig.delaxes(ax.flat[-1])
    # plt.tight_layout()


def temp3():
    raise NotImplementedError
    chani = 5
    # compare prediction and actual data around omission trials
    stim_st     = behavior.stimulus_presentations.start_time[behavior.stimulus_presentations.active].values
    omission_st = stim_st[behavior.stimulus_presentations.omitted[behavior.stimulus_presentations.active].values.astype(bool)]

    omission_error = np.zeros((len(omission_st), nbins, len(bands)+1))
    omission_data  = np.zeros((len(omission_st), nbins, len(bands)+1))

    for i in range(len(omission_st)):
        start = np.argmin(np.abs(spikes_obj.timestamps - (omission_st[i]-.25)))
        omission_error[i,:,:] = abs_error[start:start+nbins,:]
        omission_data[i,:,:]  = data[start:start+nbins,5,:]
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(30, 30))

    for bandi in range(len(bands)+1):
        ax.flat[bandi].imshow(omission_error[:,:,bandi], aspect='auto', cmap='bwr',
                            extent=[-0.25, 0.5, 0, len(stim_st)],vmin=0, vmax=2)
        avr_error = np.mean(omission_error[:,:,bandi], axis=0)
        ax.flat[bandi].plot(t, 2000 + avr_error * 800,   'w', label='average error', linewidth=3)
        avr_data = np.mean(omission_data[:,:,bandi], axis=0)
        ax.flat[bandi].plot(t, 2000 + avr_data * 800, 'k', label='average data', linewidth=3)
        
        ax.flat[bandi].set_xlabel('Time from omission (s)')
        # ax.flat[bandi].set_aspect(1.0/ax.flat[bandi].get_data_ratio(), adjustable='box')
        if bandi==0:
            ax.flat[bandi].set_ylabel('Trial')
            ax.flat[bandi].set_title('broaband')
            ax.flat[bandi].legend(loc = 'upper left')
        else:
            ax.flat[bandi].set_title('Band: ' + str(bands[bandi-1]) + ' Hz')

    # add colorbar to the last subplot
    ax.flat[-1].imshow(np.zeros((1,1)), cmap='bwr', extent=[-0.25, 0.5, 0, len(stim_st)],vmin=0, vmax=2)
    # turn off the axis
    ax.flat[-1].axis('off')
    # add colorbar
    cbar = fig.colorbar(ax.flat[-1].images[0], ax=ax.flat[-1])
    # fig.delaxes(ax.flat[-1])
    # plt.tight_layout()

def temp4():
    raise NotImplementedError
    # first 30 seconds
    spikes = spikes[:750*10, :]
    lfp = lfp[:750*10, :]
    # Reshape the data into trials
    X_reshaped = np.reshape(spikes, (10, seqlength, spikes.shape[1]))
    y_pred = model(torch.from_numpy(X_reshaped).float().to(device))
    # reshape the data back to the original shape
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1],))
    
    fig, ax = plt.subplots(1,1, figsize=(16, 5))
    ax.plot(np.linspace(0,10,int(10/bin_size)), lfp[:int(10/bin_size)],'k',label='LFP')
    ax.plot(np.linspace(0,10,int(10/bin_size)), y_pred[:int(10/bin_size)],'r',label='Transformer Prediction')
    ax.set_xlabel('Time (s)')
    ax.set_xlim([0,10])
    ax.set_ylabel('Z-scored LFP')
    ax.legend(loc='upper left', frameon=False)
    r2 = r2_score(lfp, y_pred)
    ax.set_title(f'R2 Score: {r2:.4f}')
    plt.show()