import numpy as np
import torch
import torch.nn as nn

import config
from approx_param import approximation_parameters


class SequenceIG:
    """
    History-aware integrated gradients for causal sequence models (CNN-LSTM, CNN, LSTM).

    For an input window x of L bins x U units ending at bin t, the target is the model's prediction
    averaged over the last k bins of the window (k = target_bins; k = 1 is the prediction at t):

        f(x) = mean_{j < k} y_{L-1-j}(x)

    computed by running the model over the window from its initial state, exactly as for the last
    bins of a training sequence when L = config.SEQLENGTH. Integrated gradients along the straight
    path from the baseline b to x, with the gradient taken through the conv, the recurrence and the
    readout:

        A[l, u] = (x[l, u] - b[u]) * sum_s w_s * df/dx[l, u] at (b + alpha_s (x - b))

    credits every unit at every lag l of the window. Completeness: sum_{l,u} A[l, u] = f(x) - f(b),
    up to quadrature error, returned as delta. With a zero baseline only bins with spikes receive
    attribution; a mean-rate baseline also credits silence.
    """
    def __init__(self, model, n_steps=config.IG_N_STEPS, method='gausslegendre', max_sequences=config.IG_MAX_SEQUENCES):
        self.model = model
        self.n_steps = n_steps
        step_sizes, alphas = (f(n_steps) for f in approximation_parameters(method))
        self.step_sizes = torch.tensor(step_sizes, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        # windows per batch, so that windows x n_steps sequences go through the model at once
        self.windows_per_batch = max(1, max_sequences // n_steps)

    def _f(self, x, target_bins):
        out = self.model(x)
        out = out[0] if isinstance(out, tuple) else out          # recurrent models also return their state
        return out.reshape(x.shape[0], x.shape[1])[:, -target_bins:].mean(dim=1)

    def attribute(self, windows, baseline=0.0, target_bins=1):
        """
        windows: (n, L, U) tensor on the model's device; baseline: scalar or (U,) tensor.
        Returns CPU tensors: attributions (n, L, U), f(x) (n,), f(b) (n,), completeness error delta (n,).
        """
        device = windows.device
        baseline = torch.as_tensor(baseline, dtype=windows.dtype, device=device)
        step_sizes = self.step_sizes.to(device).view(-1, 1, 1, 1)
        alphas = self.alphas.to(device).view(-1, 1, 1, 1)
        # Evaluation mode everywhere (fixed BatchNorm statistics, no dropout), except recurrent layers:
        # cuDNN only supports the RNN backward pass in training mode, which for a 1-layer LSTM
        # (no inter-layer dropout) computes exactly the same function.
        modes = {m: m.training for m in self.model.modules()}
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.RNNBase):
                m.train()
        out = {k: [] for k in ('attr', 'f_x', 'f_b', 'delta')}
        try:
            start = 0
            while start < windows.shape[0]:
                x = windows[start:start + self.windows_per_batch]
                try:
                    attr, f_x, f_b = self._attribute_batch(x, baseline, alphas, step_sizes, target_bins)
                    out_of_memory = False
                except torch.cuda.OutOfMemoryError:
                    if self.windows_per_batch == 1:
                        raise
                    out_of_memory = True
                if out_of_memory:
                    # The batch does not fit on this GPU: halve it and retry (results do not depend on
                    # the batch size); the smaller size is kept for the rest of this object's use.
                    # Done outside the except block, whose traceback still holds the failed batch's tensors.
                    self.windows_per_batch = max(1, self.windows_per_batch // 2)
                    torch.cuda.empty_cache()
                    print(f'SequenceIG: out of GPU memory, retrying with {self.windows_per_batch} windows per batch',
                          flush=True)
                    continue
                out['attr'].append(attr.cpu())
                out['f_x'].append(f_x.cpu())
                out['f_b'].append(f_b.cpu())
                out['delta'].append((attr.sum(dim=(1, 2)) - (f_x - f_b)).cpu())
                start += x.shape[0]
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        finally:
            for m, training in modes.items():
                m.training = training
        return tuple(torch.cat(out[k]) for k in ('attr', 'f_x', 'f_b', 'delta'))

    def _attribute_batch(self, x, baseline, alphas, step_sizes, target_bins):
        """Attributions, f(x) and f(b) for one batch of windows (all on the GPU, detached)."""
        b = baseline.expand_as(x)
        n, L, U = x.shape
        path = (b + alphas * (x - b)).reshape(self.n_steps * n, L, U).requires_grad_()
        grads = torch.autograd.grad(self._f(path, target_bins).sum(), path)[0]
        avg_grad = (grads.reshape(self.n_steps, n, L, U) * step_sizes).sum(dim=0)
        attr = ((x - b) * avg_grad).detach()
        del path, grads, avg_grad
        with torch.no_grad():
            return attr, self._f(x, target_bins), self._f(b, target_bins)
