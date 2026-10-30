"""
Behavioural and stimulus covariates on the spike-bin grid, for the baselines and the
stimulus-clock controls (reviewer requests: running, pupil and stimulus timing as predictors).

All functions take the left edges of the spike bins (pre_process_spikes.timestamps) and return
one value per bin.

  behaviour   running_speed (cm/s) and pupil_area, interpolated to bin centres (blinks and other
              missing samples are interpolated over); licks and rewards, counts per bin
  stimulus    onset impulses: one column per image (non-omitted, non-change presentations),
              'change' (change presentations, any image) and 'omitted'
"""
import numpy as np
import pandas as pd

import config


def bin_edges(bin_starts, bin_size=config.BIN_SIZE):
    return np.append(bin_starts, bin_starts[-1] + bin_size)


def interpolate_to_bins(timestamps, values, bin_starts, bin_size=config.BIN_SIZE):
    """
    A continuous signal at the bin centres, linearly interpolated over missing (NaN) samples and
    held at the first/last valid value outside its range. All NaN if there are fewer than 2 valid samples.
    """
    timestamps, values = np.asarray(timestamps, float), np.asarray(values, float)
    ok = np.isfinite(timestamps) & np.isfinite(values)
    if ok.sum() < 2:
        return np.full(len(bin_starts), np.nan, dtype=np.float32)
    return np.interp(bin_starts + bin_size / 2, timestamps[ok], values[ok]).astype(np.float32)


def count_events(times, bin_starts, bin_size=config.BIN_SIZE):
    """Number of events (lick, reward or onset times, s) in each bin."""
    return np.histogram(np.asarray(times, float), bins=bin_edges(bin_starts, bin_size))[0].astype(np.float32)


def behaviour(running_speed, eye_tracking, licks, rewards, bin_starts):
    """
    Behavioural covariates per bin (bins x 4 DataFrame), from the AllenSDK session tables:
    running_speed (timestamps, speed), eye_tracking (timestamps, pupil_area, likely_blink; None if
    the session has none), licks (timestamps) and rewards (timestamps).
    """
    out = pd.DataFrame({'running_speed': interpolate_to_bins(running_speed.timestamps, running_speed.speed, bin_starts)})
    if eye_tracking is None or len(eye_tracking) == 0:
        out['pupil_area'] = np.nan
    else:
        pupil = eye_tracking.pupil_area.to_numpy(float).copy()
        if 'likely_blink' in eye_tracking:
            pupil[eye_tracking.likely_blink.to_numpy(bool)] = np.nan
        out['pupil_area'] = interpolate_to_bins(eye_tracking.timestamps, pupil, bin_starts)
    out['licks'] = count_events(licks.timestamps, bin_starts)
    out['rewards'] = count_events(rewards.timestamps, bin_starts)
    return out


def stimulus_onsets(presentations, bin_starts):
    """
    Onset impulses per bin (bins x columns DataFrame) for the given stimulus presentations:
    'image_<name>' for each image's non-change presentations, 'change' and 'omitted'.
    """
    image = presentations.image_name.astype(str)
    omitted = (image == 'omitted').to_numpy()
    change = presentations.is_change.fillna(False).to_numpy(bool) & ~omitted
    columns = {f'image_{name}': count_events(presentations.start_time[(image == name) & ~change], bin_starts)
               for name in sorted(set(image[~omitted]))}
    columns['change'] = count_events(presentations.start_time[change], bin_starts)
    columns['omitted'] = count_events(presentations.start_time[omitted], bin_starts)
    return pd.DataFrame(columns)
