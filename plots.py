import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_test_loss(models, X_test, y_test, probe_obj):
    num_channels = len(probe_obj.chans)
    bands = probe_obj.bands
    scoresTest = np.zeros((num_channels, len(bands)+1))
    for chani in range(num_channels):
        for bandi in range(len(bands)+1):
            yHat_test  = models[chani, bandi].evaluate(X_test)
            # Suppose y_test is in the shape of (num_trials*seqlength, num_channels, num_bands)
            r2_test  = r2_score(y_test[:, chani, bandi].cpu().detach().numpy(), yHat_test)
            # r2_test  = r2_score(y_test[:, :, chani, bandi].cpu().detach().numpy(), yHat_test)
            scoresTest[chani, bandi] = r2_test
            
    # plot as a matrix
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    # make the plot square
    cax = ax.imshow(scoresTest, aspect='equal', cmap='viridis')
    ax.set_xlabel('Band')
    ax.set_ylabel('Channel')
    ax.set_title('Model score for all channels and bands')
    ax.set_yticks(range(num_channels))
    ax.set_yticklabels(probe_obj.chans['dorsal_ventral_ccf_coordinate'], rotation=45)
    ax.set_xticks(range(len(bands)+1))
    ax.set_xticklabels(['Broadband'] + [str(bands[i]) for i in range(len(bands))], rotation=45, ha='right')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    cbar = fig.colorbar(cax)
    cbar.set_label('R2')
    cax.set_clim(-.4, .7);

def plot_lfp_prediction(models, X_test, y_test, bands, chani):
    fig, ax = plt.subplots(int(np.ceil((len(bands)+1)/2)),2, figsize=(16, 16))
    bin_size = 0.004
    for bandi in range(len(bands)+1):
        yHat_test = models[chani, bandi].evaluate(X_test)
        ax.flat[bandi].plot(np.linspace(0,5,int(5/bin_size)), y_test.cpu().detach().numpy()[:int(5/bin_size), chani, bandi],'k',label='LFP')
        ax.flat[bandi].plot(np.linspace(0,5,int(5/bin_size)), yHat_test[:int(5/bin_size)],'r',label='LSTM')
        if bandi == 0:
            ax.flat[bandi].set_title('Channel ' + str(chani) + ' Broadband')
        else:
            ax.flat[bandi].set_title('Channel ' + str(chani) + ' Band: ' + str(bands[bandi-1]) + ' Hz')
    fig.delaxes(ax.flat[-1])