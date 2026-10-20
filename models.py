"""
Models that predict one LFP channel/band from binned spikes, and the process_model training wrapper.

Layer attribute names (lstm, batchnorm, out, conv1d, ...) are part of the saved state_dicts, so
renaming them breaks loading of already-trained models.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

import config


def build_lstm(input_size, device, seqlength=config.SEQLENGTH):
    """The project's standard per-channel LSTM (architecture in config.LSTM_KWARGS), ready to train or load."""
    return process_model(LSTMnet(input_size, seqlength=seqlength, **config.LSTM_KWARGS),
                         torch.nn.MSELoss(), device)


class CausalConv1d(nn.Module):
    """
    Conv1d over time that only sees the present and past: (batch, time, channels) in and out.

    The input is padded with kernel_size - 1 zeros on the left only, so the output at bin t
    depends on bins t - kernel_size + 1 ... t and never on later bins.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = nn.functional.pad(x.permute(0, 2, 1), (self.left_pad, 0))   # Conv1d wants (batch, channels, time)
        return self.conv(x).permute(0, 2, 1)


class LSTMnet(nn.Module):
    """
    (batch, time, units) -> (batch, time, 1): optional causal Conv1d front end, then an LSTM and a
    linear readout at every time step. Every output depends only on current and past inputs.

    forward() also returns the final (h, c) and accepts them back (used by the step-by-step
    InstantaneousIG, which is only valid without the conv front end).
    """
    def __init__(self, input_size, num_hidden, num_layers, seqlength,
                 batchNorm=False, conv=False, kernel_size=3, out_channels=16, verbose=False):
        super().__init__()
        self.verbose = verbose      # print tensor shapes in forward()
        self.batchNorm = batchNorm
        self.conv = conv
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.seqlength = seqlength

        if self.conv:
            # temporal feature extraction and dimensionality reduction: units -> out_channels
            self.conv1d = CausalConv1d(input_size, out_channels, kernel_size)
            self.input_size = out_channels   # the LSTM sees the conv channels instead of the units

        self.lstm = nn.LSTM(self.input_size, num_hidden, num_layers, batch_first=True)
        # created even when batchNorm=False so state_dicts have the same keys either way
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.out = nn.Linear(num_hidden, 1)

    def forward(self, x, h=None, c=None):
        if self.verbose:
            print(f'Input: {list(x.shape)}')

        if self.conv:
            x = self.conv1d(x)
            if self.verbose:
                print(f'Output for Conv1d: {list(x.shape)}')

        if self.batchNorm:
            # normalise each feature over all (batch, time) samples
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = self.batchnorm(x.reshape(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)

        if (c is None) and (h is None):
            y, hidden = self.lstm(x)
        else:
            y, hidden = self.lstm(x, (h, c))

        if self.verbose:
            print(f'RNN-out: {list(y.shape)}')
            print(f'RNN-hidden: {list(hidden[0].shape)}')
            print(f'RNN-cell: {list(hidden[1].shape)}')

        o = self.out(y)

        if self.verbose:
            print(f'Output: {list(o.shape)}')
        return o, hidden


class GRUnet(nn.Module):
    """GRU counterpart of LSTMnet (no conv option); forward returns (output, hidden)."""
    def __init__(self, input_size, num_hidden, num_layers, seqlength, batchNorm=False, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.batchNorm = batchNorm
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.seqlength = seqlength
        self.gru = nn.GRU(input_size, num_hidden, num_layers, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(input_size)
        self.out = nn.Linear(num_hidden, 1)

    def forward(self, x, h=None):
        if self.verbose:
            print(f'Input: {list(x.shape)}')

        if self.batchNorm:
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = self.batchnorm(x.reshape(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)

        if h is None:
            y, hidden = self.gru(x)
        else:
            y, hidden = self.gru(x, h)

        if self.verbose:
            print(f'RNN-out: {list(y.shape)}')
            print(f'RNN-hidden: {list(hidden.shape)}')

        o = self.out(y)

        if self.verbose:
            print(f'Output: {list(o.shape)}')
        return o, hidden


class process_model:
    """Training and evaluation wrapper around a torch model (LSTMnet, GRUnet or SpikingTransformer)."""
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.seqlength = model.seqlength
        self.trained = False

    def train(self, train_dataloader, test_dataloader, numepochs, lr=.001, weight_decay=0.001):
        """Adam training; returns per-epoch mean train and test losses. No early stopping."""
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses = np.zeros(numepochs)
        test_losses = np.zeros(numepochs)

        for epochi in range(numepochs):
            self.model.train()
            batchlosses = []
            for X, y in train_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                yHat = self.model(X)
                if isinstance(yHat, tuple):     # recurrent models also return their hidden state
                    yHat = yHat[0]
                loss = self.criterion(torch.squeeze(y), torch.squeeze(yHat))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batchlosses.append(loss.item())
            train_losses[epochi] = np.mean(batchlosses)

            self.model.eval()
            batchlosses = []
            with torch.no_grad():
                for X, y in test_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    yHat = self.model(X)
                    if isinstance(yHat, tuple):
                        yHat = yHat[0]
                    batchlosses.append(self.criterion(torch.squeeze(y), torch.squeeze(yHat)).item())
            test_losses[epochi] = np.mean(batchlosses)

        self.trained = True
        return train_losses, test_losses

    def evaluate(self, x_data, batch_size=1000, return_hidden=False):
        """
        Predict for 2D (time x units) or 3D (sequences x seqlength x units) input.

        2D input is truncated to a whole number of sequences. Sequences are run batch_size at a
        time to bound GPU memory. Returns a 1D CPU tensor of predictions (time,), plus the final
        hidden state per sequence if return_hidden (recurrent models only).
        """
        self.model.eval()
        self.model.to(self.device)

        if not torch.is_tensor(x_data):
            x_data = torch.tensor(x_data, dtype=torch.float32)

        if len(x_data.shape) == 2:
            num_trials = x_data.shape[0] // self.seqlength
            x_data = x_data[:num_trials * self.seqlength].reshape(num_trials, self.seqlength, x_data.shape[1])
        num_trials = x_data.shape[0]
        L = x_data.shape[0] * x_data.shape[1]

        recurrent = isinstance(self.model, (LSTMnet, GRUnet))
        yHat_list = []
        hidden_list = []
        with torch.no_grad():
            for i in range(0, num_trials, batch_size):
                x_batch = x_data[i:i + batch_size].to(self.device)
                if recurrent:
                    yHat_batch, hidden_batch = self.model(x_batch)
                    if return_hidden:
                        hidden_list.append(hidden_batch)
                else:
                    yHat_batch = self.model(x_batch)
                yHat_list.append(yHat_batch.cpu())
                del x_batch, yHat_batch
                if self.device_is_cuda():
                    torch.cuda.empty_cache()

        yHat = torch.reshape(torch.cat(yHat_list, dim=0), (L,))
        if not return_hidden:
            return yHat
        if not hidden_list:
            return yHat, None
        if isinstance(hidden_list[0], tuple):   # LSTM: (h, c), each (num_layers, batch, hidden)
            hidden = (torch.cat([h[0] for h in hidden_list], dim=1), torch.cat([h[1] for h in hidden_list], dim=1))
        else:                                   # GRU: (num_layers, batch, hidden)
            hidden = torch.cat(hidden_list, dim=1)
        return yHat, hidden

    # kept for existing callers; evaluate() is already batched
    evaluate_in_batches = evaluate

    def device_is_cuda(self):
        return torch.device(self.device).type == 'cuda'

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.trained = True
        return self.model


class LinearRegressionModel:
    """Ordinary least squares from the units in a bin to the LFP in the same bin (no temporal context)."""
    def __init__(self, input_size):
        self.model = LinearRegression()
        self.input_size = input_size
        self.trained = False

    def train(self, train_dataloader, test_dataloader, numepochs=1, lr=None, weight_decay=None):
        """
        Fit once on all training data. The per-epoch loss arrays have the same shape as
        process_model.train() returns; the values are identical across "epochs".
        """
        # one pass over the (shuffled) loader so X and y stay paired
        batches = [(X.reshape(-1, self.input_size).cpu().numpy(), y.reshape(-1).cpu().numpy())
                   for X, y in train_dataloader]
        X = np.concatenate([b[0] for b in batches])
        y = np.concatenate([b[1] for b in batches])
        self.model.fit(X, y)
        self.trained = True
        train_loss = np.mean((y - self.model.predict(X)) ** 2)
        test_loss = self.evaluate_loss(test_dataloader)
        return np.full(numepochs, train_loss), np.full(numepochs, test_loss)

    def evaluate(self, x_data):
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.reshape(-1, self.input_size).cpu().numpy()
        elif isinstance(x_data, np.ndarray):
            x_data = x_data.reshape(-1, self.input_size)
        return self.model.predict(x_data)

    def evaluate_loss(self, dataloader):
        """Mean squared error over all samples in the dataloader."""
        total_loss = 0
        num_samples = 0
        for X, y in dataloader:
            X = X.reshape(-1, self.input_size).cpu().numpy()
            y = y.reshape(-1).cpu().numpy()
            total_loss += np.sum((y - self.model.predict(X)) ** 2)
            num_samples += y.shape[0]
        return total_loss / num_samples

    def save_model(self, filename):
        np.savez(filename, coef=self.model.coef_, intercept=self.model.intercept_)

    def load_model(self, filename):
        data = np.load(filename)
        self.model.coef_ = data['coef']
        self.model.intercept_ = data['intercept']
        self.trained = True
        return self.model


class SpikingTransformer(nn.Module):
    """
    Transformer encoder from spikes to LFP, with a learnable positional embedding and an
    optional causal Conv1d after the embedding. With causal=True (default) each position attends
    only to itself and earlier positions, so like the recurrent models the prediction at time t
    uses no spikes after t.
    """
    def __init__(self, input_size, seqlength=750, embedding_dim=64, num_heads=8, num_layers=4, dropout=0.1,
                 use_conv=False, conv_kernel_size=3, conv_out_channels=16, causal=True):
        super(SpikingTransformer, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seqlength = seqlength
        self.use_conv = use_conv
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels

        # project spike counts into the model dimension
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)

        self.causal = causal
        if self.use_conv:
            self.conv1d = CausalConv1d(embedding_dim, conv_out_channels, conv_kernel_size)
            self.embedding_dim = conv_out_channels   # downstream layers see the conv channels

        # one learnable position vector per bin; inputs longer than seqlength are not supported
        self.positional_embedding = nn.Embedding(self.seqlength, self.embedding_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        self.regression_head = nn.Linear(self.embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x:  Input spike counts, shape (batch_size, sequence_length, num_neurons)
        Returns:
            lfp_predictions: Predicted LFP values, shape (batch_size, sequence_length)
        """
        batch_size, seq_len, _ = x.shape

        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        # BatchNorm1d normalises dim 1, so put the embedding dimension there
        embedded = self.embedding_bn(embedded.permute(0, 2, 1)).permute(0, 2, 1)
        embedded = self.dropout(embedded)

        if self.use_conv:
            embedded = self.conv1d(embedded)

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        embedded = embedded + self.positional_embedding(positions)

        # -inf above the diagonal: position t cannot attend to positions after t
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device) if self.causal else None
        transformer_output = self.transformer_encoder(embedded, mask=mask)  # (batch_size, sequence_length, embedding_dim)
        return self.regression_head(transformer_output).squeeze(-1)  # (batch_size, sequence_length)
