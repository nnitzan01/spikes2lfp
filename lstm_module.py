import sys 
import numpy as np 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LSTMnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, print=False):
        super().__init__()
        self.print = print
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, num_hidden, num_layers, batch_first=False)
        self.batchnorm = nn.BatchNorm1d(num_hidden)
        self.out = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if self.print:
            print(f'Input: {list(x.shape)}')
        y, hidden = self.lstm(x)
        if self.print:
            print(f'RNN-out: {list(y.shape)}')
            print(f'RNN-hidden: {list(hidden[0].shape)}')
            print(f'RNN-cell: {list(hidden[1].shape)}')
        
        # Reshape for batch normalization
        batch_size = y.shape[1]  # Assuming batch_first=False
        seq_len = y.shape[0]
        y_reshaped = y.view(seq_len * batch_size, -1)

        # Apply batch normalization
        y_normalized = self.batchnorm(y_reshaped)
        
        # Reshape back to original shape
        y_normalized = y_normalized.view(seq_len, batch_size, -1)
        
        o = self.out(y_normalized)
        if self.print:
            print(f'Output: {list(o.shape)}')
        return o
    
    
class process_model:
    def __init__(self, model, seqlength, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.seqlength = seqlength
        self.trained = False
            
    def train(self, train_dataloader, test_dataloader, numepochs, lr=.001, weight_decay=0.001):
        self.model.train()
        lossfun = self.criterion 
        input_size = train_dataloader.dataset[0][0].shape[1]
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # losses = np.zeros(numepochs)
        train_losses = np.zeros(numepochs)
        test_losses = np.zeros(numepochs)
        
        for epochi in range(numepochs):
            self.model.train()
            batchlosses = []
            for X,y in train_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                # forward pass and loss
                yHat = self.model(X)
                loss = lossfun(torch.squeeze(y), torch.squeeze(yHat))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batchlosses.append(loss.item())
                
            # average losses from this epoch
            train_losses[epochi] = np.mean(batchlosses)
            # msg = f'Finished epoch {epochi+1}/{numepochs} '
            # sys.stdout.write('\r' + msg)
            
            self.model.eval()
            batchlosses = []
            with torch.no_grad():
                for X,y in test_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    yHat = self.model(X)
                    loss = lossfun(torch.squeeze(y), torch.squeeze(yHat))
                    batchlosses.append(loss.item())
            test_losses[epochi] = np.mean(batchlosses)
            
        # flag model as "trained"
        self.trained = True
        return train_losses, test_losses

    def evaluate(self,x_data):
        self.model.eval()
        L = x_data.shape[0]
        N = L - np.mod(L, self.seqlength)
        if N < L:
            # pad the data to be a multiple of the sequence length
            pad_shape = (L-N, x_data.shape[1])
            padding = torch.zeros(pad_shape, dtype=x_data.dtype, device=x_data.device)
            x_data = torch.cat((x_data, padding), dim=0)
            
        input_size = x_data.shape[1]
        y_hat = np.zeros(x_data.shape[0])
        with torch.no_grad():
            for timei in range(0, x_data.shape[0], self.seqlength):
                # Corrected Reshape:
                X = x_data[timei:timei+self.seqlength, :].unsqueeze(1).to(self.device) #add an extra dimension for batch_size
                yy = self.model(X)
                y_hat[timei:timei+self.seqlength] = np.squeeze(yy.cpu().numpy())
        y_hat = y_hat[:L]
        return y_hat
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        # print a message if model wasn't trained
        if not hasattr(self, 'trained'):
            sys.stdout.write('Model was not yet trained.\n')
        return self.model
    
def chunk_and_reshape(spikes, lfp, seqlength, test_size=0.2, random_state=42):
    """
    Chunks the spike and LFP data into equal segments, reshapes them,
    and splits them into training and testing sets.

    Args:
        spikes: NumPy array of shape (num_timepoints, num_neurons) representing spiking data.
        lfp: NumPy array of shape (num_timepoints, num_lfp_channels) representing LFP data.
             If LFP is single channel, should be (num_timepoints, 1).
        seqlength: The length of each chunk (window size).
        test_size: The proportion of data to use for the test set.
        random_state: The random state for the train_test_split function.

    Returns:
        X_train, X_test, y_train, y_test: NumPy arrays representing the training and testing sets
                                         for the spikes (X) and LFP (y) data.
    """

    num_trials = int(lfp.shape[0] / seqlength)

    # Truncate spikes and LFP data to be multiples of seqlength
    spikes = spikes[:num_trials * seqlength, :]
    lfp = lfp[:num_trials * seqlength, :]

    # Reshape the data into trials
    X_reshaped = np.reshape(spikes, (num_trials, seqlength, spikes.shape[1]))
    lfp_reshaped = np.reshape(lfp, (num_trials, seqlength, lfp.shape[1]))

    # Split into training and testing sets at the trial level
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, lfp_reshaped, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

class LinearRegressionModel:
    def __init__(self, input_size):
        self.model = LinearRegression()
        self.input_size = input_size
        self.trained = False

    def train(self, train_dataloader, test_dataloader, numepochs=1, lr=None, weight_decay=None):
        # Linear Regression is trained in batches
        train_losses = np.zeros(numepochs)
        test_losses = np.zeros(numepochs)

        for epochi in range(numepochs):
            batch_losses = []
            for X, y in train_dataloader:
                X = X.view(-1, self.input_size).cpu().numpy()
                y = y.view(-1).cpu().numpy()
                self.model.fit(X, y)
                y_hat = self.model.predict(X)
                batch_loss = np.mean((y - y_hat) ** 2)
                batch_losses.append(batch_loss)
            train_losses[epochi] = np.mean(batch_losses)
            test_losses[epochi] = self.evaluate_loss(test_dataloader)

        self.trained = True
        return train_losses, test_losses

    def evaluate(self, x_data):
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.view(-1, self.input_size).cpu().numpy()
        elif isinstance(x_data, np.ndarray):
            x_data = x_data.reshape(-1, self.input_size)

        y_hat = self.model.predict(x_data)
        return y_hat

    def evaluate_loss(self, dataloader):
        total_loss = 0
        num_samples = 0
        for X, y in dataloader:
            X = X.view(-1, self.input_size).cpu().numpy()
            y = y.view(-1).cpu().numpy()
            y_hat = self.model.predict(X)
            loss = np.mean((y - y_hat) ** 2)  # MSE loss
            total_loss += loss * y.shape[0]
            num_samples += y.shape[0]
        return total_loss / num_samples

    def save_model(self, filename):
        # For simplicity, we'll just save the model's coefficients.
        np.savez(filename, coef=self.model.coef_, intercept=self.model.intercept_)

    def load_model(self, filename):
        data = np.load(filename)
        self.model.coef_ = data['coef']
        self.model.intercept_ = data['intercept']
        self.trained = True
        if not self.trained:
            sys.stdout.write('Model was not yet trained.\n')
        return self.model