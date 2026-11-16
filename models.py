"""
Models that predict one LFP channel/band from binned spikes, and the process_model training wrapper.

Layer attribute names (lstm, batchnorm, out, conv1d, ...) are part of the saved state_dicts, so
renaming them breaks loading of already-trained models.
"""
import numpy as np
import torch
import torch.nn as nn

import config


def build_lstm(input_size, device, seqlength=config.SEQLENGTH):
    """The project's standard per-channel model (architecture in config.LSTM_KWARGS) on `device`, ready to train or load."""
    return process_model(LSTMnet(input_size, seqlength=seqlength, **config.LSTM_KWARGS).to(device),
                         torch.nn.MSELoss(), device)


ARCHITECTURES = ['cnn_lstm', 'lstm', 'cnn']


def build_model(architecture, input_size, device, seqlength=config.SEQLENGTH):
    """
    'cnn_lstm' (the standard model, build_lstm), 'lstm' (the same without the conv front end) or 'cnn'
    (CausalCNN: the same front end and dilated causal convolutions instead of the LSTM), ready to train.
    """
    if architecture == 'cnn_lstm':
        return build_lstm(input_size, device, seqlength)
    if architecture == 'lstm':
        net = LSTMnet(input_size, seqlength=seqlength, **dict(config.LSTM_KWARGS, conv=False))
    elif architecture == 'cnn':
        net = CausalCNN(input_size, seqlength, out_channels=config.LSTM_KWARGS['out_channels'],
                        kernel_size=config.LSTM_KWARGS['kernel_size'])
    else:
        raise ValueError(f'unknown architecture {architecture!r}; choose from {ARCHITECTURES}')
    return process_model(net.to(device), torch.nn.MSELoss(), device)


class CausalConv1d(nn.Module):
    """
    Conv1d over time that only sees the present and past: (batch, time, channels) in and out.

    The input is padded with kernel_size - 1 zeros on the left only, so the output at bin t
    depends on bins t - kernel_size + 1 ... t and never on later bins.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x.permute(0, 2, 1), (self.left_pad, 0))   # Conv1d wants (batch, channels, time)
        return self.conv(x).permute(0, 2, 1)


class LSTMnet(nn.Module):
    """
    (batch, time, units) -> (batch, time, 1): optional causal Conv1d front end, then an LSTM and a
    linear readout at every time step. Every output depends only on current and past inputs.

    forward() also returns the final (h, c) and accepts them back.
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


class CausalCNN(nn.Module):
    """
    (batch, time, units) -> (batch, time, 1) without recurrence (the CNN-only baseline): LSTMnet's causal
    Conv1d front end (units -> out_channels), a 1 x 1 projection to `hidden` channels, then n_layers residual
    causal Conv1d blocks (x + ReLU(conv(x)), dilations 1, 2, 4, ...) and a linear readout at every time step.
    Receptive field 1 + (kernel_size - 1) * (2 ** (n_layers + 1) - 1) bins (n_layers 8, kernel 3: 1023 bins, 4 s).
    """
    def __init__(self, input_size, seqlength, out_channels=16, hidden=32, n_layers=8, kernel_size=3):
        super().__init__()
        self.input_size, self.seqlength = input_size, seqlength
        self.conv1d = CausalConv1d(input_size, out_channels, kernel_size)
        self.project = nn.Linear(out_channels, hidden)
        self.dilated = nn.ModuleList([CausalConv1d(hidden, hidden, kernel_size, dilation=2 ** i) for i in range(n_layers)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.project(self.conv1d(x))
        for layer in self.dilated:
            x = x + torch.relu(layer(x))
        return self.out(x)


class process_model:
    """Training and evaluation wrapper around a torch model (LSTMnet or CausalCNN)."""
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.seqlength = model.seqlength
        self.trained = False

    def train(self, train_dataloader, test_dataloader, numepochs, lr=.001, weight_decay=0.001,
              select_best=False, patience=None):
        """
        Adam training; returns per-epoch mean losses on the training and monitoring data.

        test_dataloader is only monitored. Pass a *validation* loader when select_best=True: the
        weights of the epoch with the lowest monitoring loss are restored at the end (its index is
        self.best_epoch), and with `patience` training stops after that many epochs without
        improvement (remaining loss entries are NaN).
        """
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses = np.full(numepochs, np.nan)
        test_losses = np.full(numepochs, np.nan)
        best_loss, best_state, self.best_epoch = np.inf, None, numepochs - 1

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

            if select_best:
                if test_losses[epochi] < best_loss:
                    best_loss, self.best_epoch = test_losses[epochi], epochi
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                elif patience is not None and epochi - self.best_epoch >= patience:
                    break

        if select_best and best_state is not None:
            self.model.load_state_dict(best_state)
        self.trained = True
        return train_losses, test_losses

    def n_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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

        recurrent = isinstance(self.model, LSTMnet)
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
        hidden = (torch.cat([h[0] for h in hidden_list], dim=1), torch.cat([h[1] for h in hidden_list], dim=1))   # (h, c)
        return yHat, hidden

    # kept for existing callers; evaluate() is already batched
    evaluate_in_batches = evaluate

    def device_is_cuda(self):
        return torch.device(self.device).type == 'cuda'

    def save_model(self, filename, **extra):
        """Save the weights, plus any extra entries (e.g. training history), with torch.save."""
        torch.save(dict(state_dict=self.model.state_dict(), **extra), filename)

    def load_model(self, filename):
        """Load weights saved by save_model (or a bare state_dict); returns the extra entries."""
        saved = torch.load(filename, map_location=self.device)
        state = saved.pop('state_dict') if 'state_dict' in saved else saved
        self.model.load_state_dict(state)
        self.trained = True
        return saved if state is not saved else {}
