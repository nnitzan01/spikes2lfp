import numpy as np
import pytest
import torch

from approx_param import approximation_parameters
from integrated_gradients import IntegratedGradient
from models import GRUnet, LSTMnet, SpikingTransformer


def _reference_last_time_point(net, x, n_steps):
    """Straightforward IG: for each bin t, integrate d out_t / d x_t along the path 0 -> x."""
    step_sizes, alphas = (f(n_steps) for f in approximation_parameters('gausslegendre'))
    attrs = torch.zeros_like(x)
    for step, alpha in zip(step_sizes, alphas):
        scaled = (alpha * x).clone()
        hidden = None
        for t in range(x.shape[1]):
            xt = scaled[:, t:t + 1, :].clone().requires_grad_()
            if hidden is None:
                out, hidden = net(xt)
            elif isinstance(hidden, tuple):
                out, hidden = net(xt, *hidden)
            else:
                out, hidden = net(xt, hidden)
            attrs[:, t, :] += step * torch.autograd.grad(out.sum(), xt)[0][:, 0, :]
    return (attrs * x).reshape(-1, x.shape[2])


@pytest.mark.parametrize('net', [LSTMnet(5, 6, 1, 12), GRUnet(5, 6, 2, 12)], ids=['lstm', 'gru-2-layer'])
def test_last_time_point_matches_reference(net):
    torch.manual_seed(0)
    x = (torch.rand(3, 12, 5) < 0.4).float()
    ig = IntegratedGradient(net.train(), method='last time point', seqlength=12)
    attrs = ig.run(x.reshape(-1, 5), baselines=0, n_batch=2, n_steps=6)
    np.testing.assert_allclose(attrs.numpy(), _reference_last_time_point(net, x, 6).detach().numpy(),
                               rtol=1e-5, atol=1e-6)
    assert torch.all(attrs[x.reshape(-1, 5) == 0] == 0)    # zero baseline: silent bins get no attribution


def test_transformer_is_rejected_for_last_time_point():
    with pytest.raises(ValueError):
        IntegratedGradient(SpikingTransformer(5, seqlength=12, embedding_dim=8, num_heads=2, num_layers=1),
                           method='last time point')
