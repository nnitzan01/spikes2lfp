from approx_param import approximation_parameters
from captum_utils import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
import torch
import torch.nn as nn
import numpy as np

import config
from models import SpikingTransformer

class IntegratedGradient():
    """
    Integrated gradients for the recurrent spikes -> LFP models.

    method='last time point': for every bin t, the attribution of each unit's input at t to the
    model output at t, integrating along the path from `baselines` (0 = no spikes) to the input.
    The sequence is fed one bin at a time, carrying the hidden state forward. Only the gradient
    with respect to the current bin is taken; influence through the hidden state of earlier
    bins is not attributed.

    method='moving window': gradient of the output at the last bin of each window with respect
    to all inputs in the window, optionally weighted by an exponential kernel.

    On GPU, pass the model in train() mode: cuDNN only supports the RNN backward pass in
    training mode. LSTMnet/GRUnet have no dropout, so train() does not change their output.
    """
    def __init__(self, forward_func, method='moving window', window_size=None, weight=False, tau=None, fs=250, seqlength=750):
        if method == 'last time point' and getattr(forward_func, 'conv', False):
            # stepping one bin at a time hides the conv kernel's history; use SequenceIG instead
            raise ValueError("'last time point' needs a model without a conv front end; use SequenceIG")
        if method == 'last time point' and isinstance(forward_func, SpikingTransformer):
            # feeding one bin at a time removes the attention context, so the result is not meaningful
            raise ValueError("'last time point' needs a recurrent model; SpikingTransformer is not supported")
        self.forward_func = forward_func
        self.method = method # 'last time point' or 'moving window'
        self.window_size = window_size # window size for moving window method
        self.weight = weight # whether to weight the attributions by the weights
        self.tau = tau # time constant (ms) of the exponential weighting for moving window method
        self.fs = fs # sampling frequency for moving window method, default is 250 Hz
        self.exp_func = None # exponential function for moving window method
        self.seqlength = seqlength # length of each sequence the input is split into

    def run(
        self,
        inputs,
        baselines=None,
        n_batch: int = 1,
        n_steps: int = 10,
        method: str = "gausslegendre",
    ):
        """
        Integrated gradients over a long recording, n_batch sequences at a time.

        inputs:   (total_length, input_size), split into sequences of self.seqlength
                  (trailing bins are dropped), or (n_sequences, seqlength, input_size).
                  Batches are moved to the device the input tensor is on.
        n_batch:  sequences per batch; each batch evaluates n_batch * n_steps sequences at once.

        Returns a CPU tensor:
          'last time point': (n_sequences * seqlength, input_size)
          'moving window':   (total_length - window_size + 1, input_size)
        """
        device = inputs.device
        seqlen = self.seqlength

        if self.method == "moving window":
            length = inputs.shape[0]
            if self.window_size is None:
                self.window_size = 50
            attrs = torch.zeros(length - self.window_size + 1, inputs.shape[1])
            for i in range(length - self.window_size + 1):
                inputs_slice = inputs[i:i + self.window_size, :].unsqueeze(0).to(device)
                attr_slice = self.attribute(inputs_slice, baselines, n_steps=n_steps, method=method)[0].squeeze(0)
                attrs[i, :] = attr_slice.mean(dim=0).detach().cpu()
            return attrs

        # ---- last time point mode ----
        if len(inputs.shape) == 2:
            num_trials = inputs.shape[0] // seqlen
            inputs = inputs[: num_trials * seqlen].reshape(num_trials, seqlen, inputs.shape[1])
        n_trials, _, input_size = inputs.shape

        attrs = torch.zeros(n_trials, seqlen, input_size)   # collected on the CPU
        for start in range(0, n_trials, n_batch):
            stop = min(start + n_batch, n_trials)
            attr_slice = self.attribute(inputs[start:stop].to(device), baselines,
                                        n_steps=n_steps, method=method)[0]  # (stop - start, seqlen, input_size)
            attrs[start:stop] = attr_slice.detach().cpu()
            del attr_slice
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return attrs.reshape(n_trials * seqlen, input_size)

    # MODIFIED from captum/_utils/gradient.py
    def _run_forward(self, inputs, h_c = None):
        r"""
        Run the model on one time bin, continuing from the previous bin's hidden state.

        Args:
            inputs:     tensor, inputs at time t, shape (n_steps, 1, input_size).
            h_c:        hidden state returned for t-1: an (h, c) tuple for LSTMs, a tensor for
                        GRUs, None at t=0.

        Out:
            outputs:    tensor, model output at t, shape (n_steps, 1, 1).
            h_c:        hidden state after t (None for the transformer).
        """
        if isinstance(self.forward_func, SpikingTransformer):
            return self.forward_func(inputs), None
        if h_c is None:
            return self.forward_func(inputs)
        if isinstance(h_c, tuple):          # LSTM (h, c)
            return self.forward_func(inputs, h_c[0], h_c[1])
        return self.forward_func(inputs, h_c)   # GRU hidden tensor

    # MODIFIED from captum/_utils/gradient.py
    def compute_gradients(self, inputs):
        r"""
        Gradients of the model output with respect to the inputs.

        Args:
            inputs:     tuple with one tensor, the interpolated inputs of a batch,
                        shape ([n_steps * batch, seq_length, input_size],).

        Out:
            grads:      tuple with one tensor of the same shape. For 'last time point', entry
                        [:, t, :] is d output_t / d input_t.
        """
        with torch.autograd.set_grad_enabled(True):
            # only a single input tensor is supported

            if self.method == 'moving window':
                if self.tau is None:
                    self.tau = 10 # default value
                if self.window_size is None:
                    self.window_size = inputs[0].shape[1]
                if self.exp_func is None:
                    x = np.arange(0, self.window_size)
                    t = 1 / (self.tau * 1e-3 * self.fs)
                    self.exp_func = torch.tensor(1 * np.exp(t * x) + 0)
                    self.exp_func /= torch.sum(self.exp_func)
                outputs, _ = self._run_forward(inputs[0])
                grads = torch.autograd.grad(torch.unbind(outputs[:,-1,:]), inputs)[0].squeeze()

            elif self.method == 'last time point':
                grads = torch.zeros_like(inputs[0])
                h_c = None
                for i in range(inputs[0].shape[1]):
                    # this bin's input as its own leaf tensor, shape (n_steps * batch, 1, input_size)
                    input_slice = inputs[0][:, i, :].detach().unsqueeze(1).requires_grad_()
                    outputs, h_c = self._run_forward(input_slice, h_c=h_c)
                    # samples are independent, so the gradient of the summed output gives every
                    # sample's own gradient in one backward pass
                    grads[:, i, :] = torch.autograd.grad(outputs.sum(), input_slice)[0][:, 0, :]
                    # the gradient is taken only w.r.t. the current bin, so earlier bins' graph is
                    # not needed; detaching keeps memory constant instead of growing with t
                    h_c = tuple(h.detach() for h in h_c) if isinstance(h_c, tuple) else h_c.detach()
            else:
                raise ValueError("Method not specificed or not supported.")
        return tuple([grads])

    # MODIFIED from captum/attr/_core/integrated_gradients.py
    def attribute(self,
                  inputs, 
                  baselines = None, 
                  n_steps: int = 50,
                  method: str = "gausslegendre"):
        r"""
        A wrapper function to prepare inputs and baselines for the attribution function.

        Args:
            inputs:     tensor, Total input of a batch.
                        shape (1, seq_length, input_size).
                        If forgot unsqueeze and the shape is (seq_length, input_size), it will be unsqueezed.
            baselines:  (1) int, the value of input that signifies no information. 
                        In the case of this project, it should be 0 for no neuron's firing state.
                        (2) tuple, If inputs has been z-scored for each feature, baselines should be the z-scored value
                        corresponding to the state of no-firing. Length of tuple should be equal to the number of features.
            n_steps:    int, number of steps for approximation.
            method:     str, approximation method to calculate the weights. Defaults to `gausslegendre`.

        Out:
            attributions:   tuple. 
                            Attributions of the input.
                            Shape ([1, seq_length, input_size],), where 1 is the batch size.
        """
        # Handling the inputs if it's not unsqueezed
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0)

        # format inputs and baselines into tuples
        # inputs = (inputs,)
        # baselines = (baselines,)
        inputs, baselines = _format_input_baseline(inputs, baselines)

        # check the dimensions of inputs and baselines
        _validate_input(inputs, baselines, n_steps, method)

        attributions = self._attribute(
            inputs=inputs,
            baselines=baselines,
            n_steps=n_steps,
            method=method,
        )

        return attributions
    
    # MODIFIED from captum/attr/_core/integrated_gradients.py
    def _attribute(self, inputs, baselines, n_steps = 50, method = "gausslegendre"):
        r"""
        Implementation of the integrated gradients method. To calculate the gradient of the output at each time point,
        w.r.t. the input at the same time point. Assumed to multiply the attribution with the input at the end.
        Many previously unused parameters are removed to keep it simple just for now. 

        Args:
            inputs:     tensor, Total input of a batch.
                        shape (1, seq_length, input_size).
                        If forgot unsqueeze and the shape is (seq_length, input_size), it will be unsqueezed.
            baselines:  (1) int, the value of input that signifies no information. 
                        In the case of this project, it should be 0 for no neuron's firing state.
                        (2) tuple, If inputs has been z-scored for each feature, baselines should be the z-scored value
                        corresponding to the state of no-firing. Length of tuple should be equal to the number of features.
            n_steps:    int, number of steps for approximation.
            method:     str, approximation method to calculate the weights. 
                        Supported methods by captum: `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` and `gausslegendre`. Defaults to `gausslegendre`.

        Out:
            attributions:   tuple. 
                            Attributions of the input.
                            Shape ([1, seq_length, input_size],), where 1 is the batch size.
        """
        # Obtain the function to compute the step sizes and alphas based on the method
        step_sizes_func, alphas_func = approximation_parameters(method)
        
        # Compute the step sizes and alphas for the given num_steps
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # Interpolate the baseline towards input for num_steps times, based on step sizes and alphas
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        # MODIFIED, removed formatting targets and input_additional_args
        # targets are removed because we don't need to specify a single target, all time points are used as targetes
        # input_additional_args are removed because we manually obtain and pass the hidden and cell states in the forward function
        
        # compute gradients
        grads = self.compute_gradients(inputs=scaled_features_tpl)

        # flattening grads and multiplying by step sizes
        # adding weights to each interpolation for taking the weighted sum (next step)
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # taking the weighted sum of the gradients
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # MODIFIED, assume gradient multiples inputs
        # it was assumed by Captum and also out previous calculations
        # here the choice of not multiplying is removed to keep it simple
        attributions = tuple(
            total_grad * (input - baseline)
            for total_grad, input, baseline in zip(total_grads, inputs, baselines)
        )

        # ADDED, to weight the attributions by the weights if self.weight is True
        # If method is moving window, attribution should be multipled by the exponential function
        if self.method == 'moving window':
            if self.weight:
                attributions = tuple(
                    (attribution.squeeze().T * self.exp_func.to(attribution.device)).T
                    for attribution in attributions
                )

        return attributions

class SequenceIG:
    """
    History-aware integrated gradients for causal sequence models (CNN-LSTM, causal transformer).

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
