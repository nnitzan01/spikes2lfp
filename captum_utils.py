import torch
from torch import Tensor
from approx_param import SUPPORTED_METHODS

# Taken from captum.attr._utils.common
def _format_tensor_into_tuples(inputs):
    if inputs is None:
        return None
    if not isinstance(inputs, tuple):
        assert isinstance(inputs, torch.Tensor), (
            "`inputs` must be a torch.Tensor or a tuple[torch.Tensor] "
            f"but found: {type(inputs)}"
        )
        inputs = (inputs,)
    return inputs

def _zeros(inputs):
    r"""
    Takes a tuple of tensors as input and returns a tuple that has the same
    length as `inputs` with each element as the integer 0.
    """
    return tuple(0 if input.dtype is not torch.bool else False for input in inputs)

def _format_baseline(baselines, inputs):
    if baselines is None:
        return _zeros(inputs)

    if not isinstance(baselines, tuple):
        baselines = (baselines,)

    for baseline in baselines:
        assert isinstance(
            baseline, (torch.Tensor, int, float)
        ), "baseline input argument must be either a torch.Tensor or a number \
            however {} detected".format(
            type(baseline)
        )

    return baselines

def _format_input_baseline(inputs, baselines):
    inputs = _format_tensor_into_tuples(inputs)
    baselines = _format_baseline(baselines, inputs)
    return inputs, baselines

def _reshape_and_sum(
    tensor_input, num_steps, num_examples, layer_size
):
    # Used for attribution methods which perform integration
    # Sums across integration steps by reshaping tensor to
    # (num_steps, num_examples, (layer_size)) and summing over
    # dimension 0. Returns a tensor of size (num_examples, (layer_size))
    return torch.sum(
        tensor_input.reshape((num_steps, num_examples) + layer_size), dim=0
    )

def _validate_input_basic(inputs, baselines, draw_baseline_from_distrib,):
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

def _validate_input(
    inputs,
    baselines,
    n_steps: int = 50,
    method: str = "riemann_trapezoid",
    draw_baseline_from_distrib: bool = False,
):
    _validate_input_basic(inputs, baselines, draw_baseline_from_distrib)
    assert (
        n_steps >= 0
    ), "The number of steps must be a positive integer. " "Given: {}".format(n_steps)

    assert (
        method in SUPPORTED_METHODS
    ), "Approximation method must be one for the following {}. " "Given {}".format(
        SUPPORTED_METHODS, method
    )