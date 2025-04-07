import captum
from inspect import signature

from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)

from captum._utils.common import (
    _reduce_list,
    _run_forward,
    _sort_key_list,
    _verify_select_neuron,
)

from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
    _run_forward,
)

from captum.attr._utils.approximation_methods import approximation_parameters

def _format_inputs(inputs, unpack_inputs = True):
    return (
        inputs
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and unpack_inputs
        else (inputs,)
    )

def validate_input(inputs, baselines, draw_baseline_from_distrib = False):
    assert len(inputs) == len(baselines), (
        "Input and baseline must have the same "
        "dimensions, baseline has {} features whereas input has {}.".format(
            len(baselines), len(inputs)
        )
    )

    for input, baseline in zip(inputs, baselines):
        if draw_baseline_from_distrib:
            assert (
                isinstance(baseline, (int, float))
                or input.shape[1:] == baseline.shape[1:]
            ), (
                "The samples in input and baseline batches must have"
                " the same shape or the baseline corresponding to the"
                " input tensor must be a scalar."
                " Found baseline: {} and input: {} ".format(baseline, input)
            )
        else:
            assert (
                isinstance(baseline, (int, float))
                or input.shape == baseline.shape
                or baseline.shape[0] == 1
            ), (
                "Baseline can be provided as a tensor for just one input and"
                " broadcasted to the batch or input and baseline must have the"
                " same shape or the baseline corresponding to each input tensor"
                " must be a scalar. Found baseline: {} and input: {}".format(
                    baseline, input
                )
            )

def gauss_legendre_builders():
    r"""Numpy's `np.polynomial.legendre` function helps to compute step sizes
    and alpha coefficients using gauss-legendre quadrature rule.
    Since numpy returns the integration parameters in different scales we need to
    rescale them to adjust to the desired scale.

    Gauss Legendre quadrature rule for approximating the integrals was originally
    proposed by [Xue Feng and her intern Hauroun Habeeb]
    (https://research.fb.com/people/feng-xue/).

    Returns:
        2-element tuple of **step_sizes**, **alphas**:
        - **step_sizes** (*Callable*):
                    `step_sizes` takes the number of steps as an
                    input argument and returns an array of steps sizes which
                    sum is smaller than or equal to one.

        - **alphas** (*Callable*):
                    `alphas` takes the number of steps as an input argument
                    and returns the multipliers/coefficients for the inputs
                    of integrand in the range of [0, 1]

    """
    import numpy as np
    def step_sizes(n: int):
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from 2 to 1
        return list(0.5 * np.polynomial.legendre.leggauss(n)[1])

    def alphas(n: int):
        assert n > 0, "The number of steps has to be larger than zero"
        # Scaling from [-1, 1] to [0, 1]
        return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))

    return step_sizes, alphas


def format_additional_forward_args(additional_forward_args):
    if additional_forward_args is not None and not isinstance(
        additional_forward_args, tuple
    ):
        additional_forward_args = (additional_forward_args,)
    return additional_forward_args

def _expand_additional_forward_args(additional_forward_args,n_steps,expansion_type,):
    import torch
    def _expand_tensor_forward_arg(additional_forward_arg,n_steps: int,expansion_type,):
        import torch
        if len(additional_forward_arg.size()) == 0:
            return additional_forward_arg
        if expansion_type == ExpansionTypes.repeat: #type: ignore
            return torch.cat([additional_forward_arg] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave: #type: ignore
            return additional_forward_arg.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )
    if additional_forward_args is None:
        return None

    return tuple(
        _expand_tensor_forward_arg(additional_forward_arg, n_steps, expansion_type)
        if isinstance(additional_forward_arg, torch.Tensor)
        else additional_forward_arg
        for additional_forward_arg in additional_forward_args
    )

def _expand_target(target,n_steps: int,expansion_type):
    if isinstance(target, list):
        if expansion_type == ExpansionTypes.repeat:
            return target * n_steps
        elif expansion_type == ExpansionTypes.repeat_interleave:
            expanded_target = []
            for i in target:
                expanded_target.extend([i] * n_steps)
            return cast(Union[List[Tuple[int, ...]], List[int]], expanded_target)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    elif isinstance(target, torch.Tensor) and torch.numel(target) > 1:
        if expansion_type == ExpansionTypes.repeat:
            return torch.cat([target] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave:
            return target.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )
    return target


class IntegratedGradient():
    def __init__(self, forward_func):
        self.forward_func = forward_func

    # MODIFIED
    # taken from captum/_utils/gradient.py
    def _run_forward(
    forward_func,
    inputs,
    target,
    ):
        # convert inputs to tuple if not already a tuple
        inputs = _format_inputs(inputs)
        output = forward_func(*inputs)
        
        return _select_targets(output, target)

    def compute_gradients(
        forward_fn,
        inputs,
        target_ind = None,
        additional_forward_args = None,
        ):
        import torch
        r"""
        Computes gradients of the output with respect to inputs for an
        arbitrary forward function.

        Args:

            forward_fn: forward function. This can be for example model's
                        forward function.
            input:      Input at which gradients are evaluated,
                        will be passed to forward_fn.
            target_ind: Index of the target class for which gradients
                        must be computed (classification only).
            additional_forward_args: Additional input arguments that forward
                        function requires. It takes an empty tuple (no additional
                        arguments) if no additional arguments are required
        """
        with torch.autograd.set_grad_enabled(True):
            # runs forward pass
            outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
            assert outputs[0].numel() == 1, (
                "Target not provided when necessary, cannot"
                " take gradient with respect to multiple outputs."
            )
            # torch.unbind(forward_out) is a list of scalar tensor tuples and
            # contains batch_size * #steps elements
            grads = torch.autograd.grad(torch.unbind(outputs), inputs)
        return grads

    def attribute(self, 
                  inputs, 
                  baselines = None, 
                  n_steps: int = 50,
                  method: str = "gausslegendre",
                  additional_forward_args = None):

        # format inputs and baselines into tuples
        inputs, baselines = _format_input_baseline(inputs, baselines)

        # check the dimensions of inputs and baselines
        _validate_input(inputs, baselines, n_steps, method)

        attributions = self._attribute(
            inputs=inputs,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
            )
        return attributions

    # specific implementation of the gradient function
    def _attribute(self, inputs, baselines, n_steps = 50, method = "gausslegendre", step_sizes_and_alphas = None,):
        import torch
        validate_input(inputs, baselines, n_steps, method)

        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = format_additional_forward_args(additional_forward_args)

        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )

        # MODIFIED, removed target
        # instead of using self.gradient_func, we use the modified compute_gradients function
        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        return attributions

    # wrapper function to calculate the attribution scores
    
