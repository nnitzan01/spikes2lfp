import sys 
import numpy as np 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LSTMnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, seqlength, batchNorm = False , print=False):
        super().__init__()
        self.print = print
        self.batchNorm = batchNorm
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.seqlength = seqlength
        self.lstm = nn.LSTM(input_size, num_hidden, num_layers, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(input_size)
        self.out = nn.Linear(num_hidden, 1)

    def forward(self, x, h = None, c = None):
        if self.print:
            print(f'Input: {list(x.shape)}')
            
        if self.batchNorm:
            # Reshape for batch normalization
            batch_size = x.shape[0]  # Assuming batch_first=False
            seq_len    = x.shape[1]
            x_reshaped = x.view(batch_size * seq_len, -1)
            x_normalized = self.batchnorm(x_reshaped)
            x = x_normalized.view(batch_size, seq_len, -1)    
        
        if (c is None) and (h is None):
            y, hidden = self.lstm(x)
        else:
            y, hidden = self.lstm(x, (h, c))
        
        if self.print:
            print(f'RNN-out: {list(y.shape)}')
            print(f'RNN-hidden: {list(hidden[0].shape)}')
            print(f'RNN-cell: {list(hidden[1].shape)}')

        o = self.out(y)
        
        if self.print:
            print(f'Output: {list(o.shape)}')
        return o, hidden
    
class process_model:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.seqlength = model.seqlength
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
                # if yHat is a tuple, take the first element
                if isinstance(yHat, tuple):
                    yHat = yHat[0]
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
                    if isinstance(yHat, tuple):
                        yHat = yHat[0]
                    loss = lossfun(torch.squeeze(y), torch.squeeze(yHat))
                    batchlosses.append(loss.item())
            test_losses[epochi] = np.mean(batchlosses)
            
        # flag model as "trained"
        self.trained = True
        return train_losses, test_losses
    
    def evaluate(self, x_data, return_hidden=False):
        self.model.eval()
        self.model.to(self.device)
        
        # check if the data is a tensor
        if not torch.is_tensor(x_data):
            x_data = torch.tensor(x_data, dtype=torch.float32, device=self.device)
        else:
            x_data = x_data.to(self.device)
        
        # if the data is a 2D tensor, add an extra dimension for batch_size otherwise assume it is already 3D
        if len(x_data.shape) == 2:
            L = x_data.shape[0]
            if L % self.seqlength != 0:
                x_data = x_data[:-(L % self.seqlength), :]
            L = x_data.shape[0]

            num_trials = int(L / self.seqlength)
            input_size = x_data.shape[1]
            x_data = torch.reshape(x_data, (num_trials, self.seqlength, input_size))
        else:
            L = x_data.shape[0] * x_data.shape[1]
      
        with torch.no_grad():
            if return_hidden and isinstance(self.model, LSTMnet):
                yHat, hidden = self.model(x_data)
            elif (not return_hidden) and isinstance(self.model, LSTMnet):
                yHat, _ = self.model(x_data)
            else:
                yHat = self.model(x_data)

        # Reshape the output back to (L,)
        yHat = torch.reshape(yHat, (L,)) # as we are going from (batch_size, seq_length) to (batch_size*seq_length,)
        
        if return_hidden:
            return yHat, hidden
        else:
            return yHat
    
    # def evaluate(self,x_data):
    #     self.model.eval()
    #     L = x_data.shape[0]
    #     N = L - np.mod(L, self.seqlength)
    #     if N < L:
    #         # pad the data to be a multiple of the sequence length
    #         pad_shape = (L-N, x_data.shape[1])
    #         padding = torch.zeros(pad_shape, dtype=x_data.dtype, device=x_data.device)
    #         x_data = torch.cat((x_data, padding), dim=0)
            
    #     input_size = x_data.shape[1]
    #     y_hat = np.zeros(x_data.shape[0])
    #     with torch.no_grad():
    #         for timei in range(0, x_data.shape[0], self.seqlength):
    #             # Corrected Reshape:
    #             X = x_data[timei:timei+self.seqlength, :].unsqueeze(1).to(self.device) #add an extra dimension for batch_size
    #             yy = self.model(X)
    #             y_hat[timei:timei+self.seqlength] = np.squeeze(yy.cpu().numpy())
    #     y_hat = y_hat[:L]
    #     return y_hat
    
    # def evaluate(self, x_data, overlap_factor=0.5):
    #     self.model.eval()
    #     L = x_data.shape[0]
    #     y_hat = np.zeros(L)
    #     hop_length = int(self.seqlength * (1 - overlap_factor))

    #     with torch.no_grad():
    #         for timei in range(0, L - self.seqlength + 1, hop_length):
    #             if type(self.model) is SpikingTransformer:
    #                 current_seq = x_data[timei:timei+self.seqlength, :].unsqueeze(0).to(self.device)
    #             elif type(self.model) is LSTMnet:
    #                 current_seq = x_data[timei:timei+self.seqlength, :].unsqueeze(1).to(self.device)

    #             # Check if the sequence needs padding (at the beginning)
    #             if current_seq.shape[0] < self.seqlength:
    #                 pad_length = self.seqlength - current_seq.shape[0]
    #                 if type(self.model) is SpikingTransformer:
    #                     pad_shape = (current_seq.shape[0], pad_length, current_seq.shape[2])
    #                     padding = torch.zeros(pad_shape, dtype=current_seq.dtype, device=current_seq.device)
    #                     current_seq = torch.cat((padding, current_seq), dim=1)
    #                 elif type(self.model) is LSTMnet:
    #                     pad_shape = (pad_length, current_seq.shape[1], current_seq.shape[2])
    #                     padding = torch.zeros(pad_shape, dtype=current_seq.dtype, device=current_seq.device)
    #                     current_seq = torch.cat((padding, current_seq), dim=0)

    #             yy = self.model(current_seq)
                
    #             # Assign predictions to y_hat with overlap handling
    #             if type(self.model) is SpikingTransformer:
    #                 y_hat[timei:timei+self.seqlength] = yy.cpu().numpy()
    #             elif type(self.model) is LSTMnet:
    #                 y_hat[timei:timei+self.seqlength] = yy[:,0,0].cpu().numpy()
                
    #         # handle the last chunk
    #         if timei + self.seqlength < L:
    #             if type(self.model) is SpikingTransformer:
    #                 current_seq = x_data[timei:L, :].unsqueeze(0).to(self.device)
    #                 pad_length = self.seqlength - current_seq.shape[1]
    #                 pad_shape = (current_seq.shape[0], pad_length, current_seq.shape[2])
    #                 padding = torch.zeros(pad_shape, dtype=current_seq.dtype, device=current_seq.device)
    #                 current_seq = torch.cat((current_seq, padding), dim=1)
    #             elif type(self.model) is LSTMnet:
    #                 current_seq = x_data[timei:L, :].unsqueeze(1).to(self.device)
    #                 pad_length = self.seqlength - current_seq.shape[0]
    #                 pad_shape = (pad_length, current_seq.shape[1], current_seq.shape[2])
    #                 padding = torch.zeros(pad_shape, dtype=current_seq.dtype, device=current_seq.device)
    #                 current_seq = torch.cat((current_seq, padding), dim=0)
                    
    #             yy = self.model(current_seq)
                
    #             if type(self.model) is SpikingTransformer:
    #                 y_hat[timei:L] = yy[:L-timei].cpu().numpy()
    #             elif type(self.model) is LSTMnet:
    #                 y_hat[timei:L] = yy[:L-timei,0,0].cpu().numpy()
    #     return y_hat
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        # print a message if model wasn't trained
        if not hasattr(self, 'trained'):
            sys.stdout.write('Model was not yet trained.\n')
        return self.model
    
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
    
    
class SpikingTransformer(nn.Module):
    def __init__(self, input_size, seqlength = 750, embedding_dim = 64, num_heads = 8, num_layers = 4, dropout=0.1):
        super(SpikingTransformer, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seqlength = seqlength

        # Embedding Layer:  Maps spike counts to a higher-dimensional space
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim) # Batchnorm after embedding

        # Positional Encoding (Learnable)
        self.positional_embedding = nn.Embedding(self.seqlength, embedding_dim)  # Window size is 750

        # Transformer Encoder Layers
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True # Important for PyTorch >= 1.9
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        # Regression Head
        self.regression_head = nn.Linear(embedding_dim, 1)  # Predicts a single LFP value

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x:  Input spike counts, shape (batch_size, sequence_length, num_neurons)
        Returns:
            lfp_predictions: Predicted LFP values, shape (batch_size, sequence_length)
        """
        batch_size, seq_len, _ = x.shape

        # Embedding
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        #Reshape is required before applying batch norm!
        embedded = embedded.permute(0,2,1)
        embedded = self.embedding_bn(embedded)
        embedded = embedded.permute(0,2,1)
        embedded = self.dropout(embedded)

        # Positional Encoding
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)  # (batch_size, sequence_length)
        positional_embeddings = self.positional_embedding(positions) # (batch_size, sequence_length, embedding_dim)
        embedded += positional_embeddings

        # Transformer Encoder
        transformer_output = self.transformer_encoder(embedded) # (batch_size, sequence_length, embedding_dim)

        # Regression Head
        lfp_predictions = self.regression_head(transformer_output).squeeze(-1)  # (batch_size, sequence_length)

        return lfp_predictions