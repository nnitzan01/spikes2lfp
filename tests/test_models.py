import torch

import config
import models


def test_build_lstm_matches_config_and_runs():
    model = models.build_lstm(input_size=7, device='cpu', seqlength=20)
    assert model.model.num_hidden == config.LSTM_KWARGS['num_hidden']
    assert model.model.conv == config.LSTM_KWARGS['conv']
    y = model.evaluate(torch.rand(40, 7))      # 2 sequences of 20 bins, flattened
    assert y.shape == (40,)
