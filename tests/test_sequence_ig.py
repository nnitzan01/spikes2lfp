import pytest
import torch
import torch.nn as nn

from integrated_gradients import SequenceIG
from models import CausalConv1d, LSTMnet, SpikingTransformer


class LinearCausal(nn.Module):
    """y_t = sum_{j<3} sum_u w[u, j] x[t-2+j, u] + c: IG is exact for a linear model."""
    def __init__(self, n_units):
        super().__init__()
        self.conv = CausalConv1d(n_units, 1, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


def _spiky(n, L, U, p=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, L, U, generator=g) < p).float()


def test_linear_model_attribution_is_weight_times_input_on_kernel_lags_only():
    torch.manual_seed(0)
    net = LinearCausal(4)
    x = _spiky(3, 12, 4)
    attr, f_x, f_b, delta = SequenceIG(net, n_steps=8).attribute(x)
    w = net.conv.conv.weight.detach()[0]                 # (units, kernel): lag j=0 is the oldest bin
    expected = torch.zeros_like(x)
    expected[:, -3:, :] = x[:, -3:, :] * w.T
    torch.testing.assert_close(attr, expected, rtol=1e-5, atol=1e-6)
    assert delta.abs().max() < 1e-5


@pytest.mark.parametrize('target_bins', [1, 5])
def test_cnn_lstm_completeness_and_history(target_bins):
    torch.manual_seed(0)
    net = LSTMnet(6, 8, 1, 40, conv=True, kernel_size=3, out_channels=4)
    x = _spiky(5, 40, 6)
    attr, f_x, f_b, delta = SequenceIG(net, n_steps=50).attribute(x, target_bins=target_bins)
    assert (delta.abs() <= 1e-3 * (f_x - f_b).abs() + 1e-5).all()        # sum of attributions = f(x) - f(b)
    # credit reaches spikes well before the conv kernel's 3 bins, through the recurrence
    assert attr[:, :-(3 + target_bins), :].abs().sum() > 0.05 * attr.abs().sum()
    # zero baseline: bins without spikes get no attribution
    assert torch.all(attr[x == 0] == 0)


def test_mean_baseline_credits_silence_and_keeps_completeness():
    torch.manual_seed(1)
    net = LSTMnet(6, 8, 1, 30, conv=True, kernel_size=3, out_channels=4)
    x = _spiky(4, 30, 6)
    baseline = x.mean(dim=(0, 1))                       # per-unit mean input
    attr, f_x, f_b, delta = SequenceIG(net, n_steps=50).attribute(x, baseline=baseline)
    assert attr[x == 0].abs().sum() > 0
    assert (delta.abs() <= 1e-3 * (f_x - f_b).abs() + 1e-5).all()


def test_batching_does_not_change_results_and_transformer_is_supported():
    torch.manual_seed(2)
    net = SpikingTransformer(5, seqlength=20, embedding_dim=8, num_heads=2, num_layers=1, dropout=0.0)
    x = _spiky(6, 20, 5)
    a1, *_ = SequenceIG(net, n_steps=10, max_sequences=10).attribute(x)     # one window per batch
    a2, f_x, f_b, delta = SequenceIG(net, n_steps=10, max_sequences=1000).attribute(x)
    torch.testing.assert_close(a1, a2, rtol=1e-4, atol=1e-6)
    _, f_x, f_b, delta = SequenceIG(net, n_steps=50).attribute(x)
    assert (delta.abs() <= 1e-2 * (f_x - f_b).abs() + 1e-4).all()


def test_out_of_memory_halves_the_batch_and_gives_the_same_result():
    torch.manual_seed(3)
    net = LSTMnet(5, 6, 1, 20, conv=True, kernel_size=3, out_channels=4)

    class SmallGPU(nn.Module):
        """Raises CUDA OOM whenever more than 12 sequences go through at once."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            if x.shape[0] > 12:
                raise torch.cuda.OutOfMemoryError('simulated')
            return self.inner(x)

    x = _spiky(7, 20, 5)
    expected, *_ = SequenceIG(net, n_steps=4).attribute(x)
    ig = SequenceIG(SmallGPU(net), n_steps=4, max_sequences=1000)     # starts at 250 windows per batch
    attr, f_x, f_b, delta = ig.attribute(x)
    assert ig.windows_per_batch * 4 <= 12
    torch.testing.assert_close(attr, expected, rtol=1e-5, atol=1e-7)
