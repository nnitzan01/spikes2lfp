import numpy as np
import pandas as pd

import covariates


def test_interpolation_skips_missing_samples_and_holds_ends():
    bins = np.arange(0, 1, 0.25)                    # centres 0.125, 0.375, 0.625, 0.875
    values = covariates.interpolate_to_bins([0.2, 0.4, 0.6], [1.0, np.nan, 3.0], bins, bin_size=0.25)
    np.testing.assert_allclose(values, [1.0, 1.875, 3.0, 3.0])
    assert np.isnan(covariates.interpolate_to_bins([0.1], [1.0], bins, bin_size=0.25)).all()


def test_behaviour_masks_blinks_and_counts_events():
    bins = np.arange(250) * 0.004                  # the spike-bin grid (config.BIN_SIZE)
    run = pd.DataFrame({'timestamps': [0, 1], 'speed': [0, 10]})
    eye = pd.DataFrame({'timestamps': [0, 0.5, 1], 'pupil_area': [1, 100, 1], 'likely_blink': [False, True, False]})
    licks = pd.DataFrame({'timestamps': [0.05, 0.06, 0.55]})
    rewards = pd.DataFrame({'timestamps': [0.55]})
    b = covariates.behaviour(run, eye, licks, rewards, bins)
    assert list(b.columns) == ['running_speed', 'pupil_area', 'licks', 'rewards']
    np.testing.assert_allclose(b.running_speed, 10 * (bins + 0.002), atol=1e-4)     # bin centres
    np.testing.assert_allclose(b.pupil_area, 1)                       # the blink sample is ignored
    assert b.licks.sum() == 3 and b.licks.iloc[12] == 1 and b.licks.iloc[15] == 1 and b.licks.iloc[137] == 1
    assert b.rewards.sum() == 1
    assert covariates.behaviour(run, None, licks, rewards, bins).pupil_area.isna().all()


def test_stimulus_onsets():
    stim = pd.DataFrame({'start_time': [0.0, 0.3, 0.6, 0.9], 'image_name': ['a', 'b', 'omitted', 'b'],
                         'is_change': [False, True, False, False]})
    s = covariates.stimulus_onsets(stim, np.arange(300) * 0.004)
    assert list(s.columns) == ['image_a', 'image_b', 'change', 'omitted']
    assert s.image_a.sum() == 1 and s.image_b.sum() == 1 and s.change.sum() == 1 and s.omitted.sum() == 1
    assert s.change.iloc[75] == 1 and s.image_b.iloc[225] == 1 and s.omitted.iloc[150] == 1
