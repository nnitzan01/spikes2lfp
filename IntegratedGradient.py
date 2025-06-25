from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
import numpy as np
from tqdm import tqdm
import torch
from models import SpikingTransformer

class IntegratedGradient():
    def __init__(self, forward_func, method='moving window', window_size=None, weight=False, tau=None, fs=250, seqlength=750):
        self.forward_func = forward_func
        self.method = method # 'last time point' or 'moving window'
        self.window_size = window_size # window size for moving window method
        self.weight = weight # whether to weight the attributions by the weights
        self.tau = tau # time constant of the exponential function for moving window method
        self.fs = fs # sampling frequency for moving window method, default is 250 Hz
        self.exp_func = None # exponential function for moving window method
        self.seqlength = seqlength # length of the sequence for moving window method, default is 750

    def run(self, 
            inputs, 
            baselines = None, 
            n_batch: int = 10,
            n_steps: int = 50,
            method: str = "gausslegendre"):
        r"""
        A wrapper function to run the integrated gradients method.

        Args:

            inputs:     tensor, Total input of a batch.
                        shape (total_length, input_size).
                        If method is "last time point", it will be reshaped to (total_length//750, 750, input_size)
            baselines:  (1) int, the value of input that signifies no information. 
                        In the case of this project, it should be 0 for no neuron's firing state.
                        (2) tuple, If inputs has been z-scored for each feature, baselines should be the z-scored value
                        corresponding to the state of no-firing. Length of tuple should be equal to the number of features.
            n_steps:    int, number of steps for approximation.
            method:     str, approximation method to calculate the weights. Defaults to `gausslegendre`.
        
        Out:

            attrs:      tensor, Attributions of the input.
                        shape (total_length, input_size) if method is "last time point"
                        shape (total_length - window_size + 1, input_size) if method is "moving window"
        
        """
        if self.method == 'moving window':
            length = inputs.shape[0]
            if self.window_size is None:
                self.window_size = 50
            attrs = torch.zeros(length-self.window_size+1, inputs.shape[1])
            n_slides = length - self.window_size + 1
            for i in tqdm(range(n_slides)):
                inputs_slice = inputs[i:i+self.window_size, :].unsqueeze(0)
                attrs[i, :] = torch.mean(self.attribute(inputs_slice, baselines, n_steps=n_steps, method=method)[0].squeeze(0),axis=0)
            return attrs
        else:
            # assume input is of shape (n_batch, seq_length, input_size)
            if len(inputs.shape) == 2:
                num_trials = inputs.shape[0]//self.seqlength 
                inputs = inputs[:num_trials*self.seqlength ].reshape(num_trials, self.seqlength , inputs.shape[1])
            attrs = torch.zeros_like(inputs)
            counter = 0
            iterations = inputs.shape[0] // n_batch
            for i in range(iterations):
                inputs_slice = inputs[counter:counter+n_batch, :, :]
                attrs[counter:counter+n_batch, :, :] = self.attribute(inputs_slice, baselines, n_steps=n_steps, method=method)[0].squeeze(0)
                counter += n_batch
            attrs = attrs.reshape(attrs.shape[0]*attrs.shape[1], attrs.shape[2])
            return attrs

    # MODIFIED from captum/_utils/gradient.py
    def _run_forward(self, inputs, h_c = None):
        r"""
        Computes gradient time point by time point by feeding the sliced input and 
        the hidden and cell states (if not t=0) to the forward function.

        Args:

            inputs:     tensor, sliced inputs at time t, shape 
                        (n_steps, 1, input_size).
            h_c:        tuple of hidden and cell states, shape 
                        ((n_steps, 1, hidden_size), (n_steps, 1, hidden_size)).
                        If t=0, h_c is None.
        
        Out:
            outputs:    tensor, output of the forward function, shape 
                        (n_steps, 1, 1).
            h_c:        tuple of hidden and cell states, shape 
                        ((n_steps, 1, hidden_size), (n_steps, 1, hidden_size)).
        """
        if h_c is not None:
            if len(h_c) == 2:
                outputs, h_c = self.forward_func(inputs, h_c[0], h_c[1])
                return outputs, h_c
            else:
                outputs, h_c = self.forward_func(inputs, h_c)
                return outputs, h_c
        else:
            if type(self.forward_func) is SpikingTransformer:
                outputs = self.forward_func(inputs)
                return outputs, None
            else:
                outputs, h_c = self.forward_func(inputs)        
                return outputs, h_c
        

    # MODIFIED from captum/_utils/gradient.py
    def compute_gradients(self, inputs):
        r"""
        Computes gradients of the output with respect to inputs for an
        arbitrary forward function.

        Args:

            forward_fn: forward function. In this project's case, it is the LSTM
                        model's forward function, i.e. model.model().
            input:      tuple, inside tuple: tensor. Total input of a batch. Will be sliced into time points
                        and feed into the forward function one by one.
                        shape ([n_steps, seq_length, input_size],), where 1 is the length of the tuple.

        Out:
            grads:      tuple, before casting: tensor. 
                        Gradient of the output with respect to inputs, calculated at each time point, 
                        shape ([n_steps, seq_length, input_size],).
        """
        with torch.autograd.set_grad_enabled(True):
            # assuming len(inputs) == 1 for now for the ease of implementation
            # len(inputs) == 1 means the input to IG is a single batch
            
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
                # grads = torch.zeros((inputs[0].shape[0],inputs[0].shape[1]))
                outputs, _ = self._run_forward(inputs[0])
                grads = torch.autograd.grad(torch.unbind(outputs[:,-1,:]), inputs)[0].squeeze()

            elif self.method == 'last time point':
                grads = torch.zeros_like(inputs[0])
                # iterate over time points
                for i in range(inputs[0].shape[1]):
                    # slice the input at time t and unsqueeze, shape (n_steps, 1, input_size)
                    input_slice = inputs[0][:,i,:].requires_grad_().unsqueeze(1)
                    if i == 0:
                        # first time point, no hidden and cell states
                        outputs, h_c = self._run_forward(input_slice)
                    else:
                        # subsequent time points, use the hidden and cell states
                        outputs, h_c = self._run_forward(input_slice, h_c = h_c)
                    # compute gradients using autograd, the output is a tuple of tensors of shape (n_steps, 1, input_size)
                    grads[:,i,:] = torch.autograd.grad(torch.unbind(outputs), input_slice)[0].squeeze()
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