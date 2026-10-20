import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo's top-level modules importable regardless of where pytest is launched from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config            # noqa: E402  (needs the sys.path entry above)
import preprocess_data   # noqa: E402

# --- a small synthetic session that stands in for AllenSDK + NWB data in end-to-end tests ---

N_UNITS, SESSION_LEN, N_CHANNELS = 12, 120, 2
HVS_SESSION = 1049273528      # marked as having HVS in tables/hvs_detection.csv


class FakeSession:
    def __init__(self, session_id, df, output_dir, target_areas=None):
        rng = np.random.default_rng(session_id % 1000)
        areas = np.array(['VISp', 'VISl', 'LGd', 'CA1'])
        self.units = pd.DataFrame({'structure_acronym': areas[np.arange(N_UNITS) % 4],
                                   'firing_rate': rng.uniform(1, 20, N_UNITS)},
                                  index=np.arange(500, 500 + N_UNITS))
        self.spike_times = {u: np.sort(rng.uniform(0, SESSION_LEN, rng.integers(300, 3000))) for u in self.units.index}
        starts = np.arange(2.0, SESSION_LEN - 2, 0.75)
        n = len(starts)
        self.stimulus_presentation = pd.DataFrame({
            'start_time': starts, 'active': starts < 60, 'is_change': np.arange(n) % 7 == 0,
            'image_name': np.where(np.arange(n) % 10 == 0, 'omitted', 'im')})
        self.active = self.stimulus_presentation[self.stimulus_presentation.active]
        self.active_times, self.spontaneous_times, self.passive_times = [0, 60], [60, 75], [75, SESSION_LEN]
        self.channels = pd.DataFrame({'structure_acronym': ['VISp'] * N_CHANNELS + ['VISl'],
                                      'dorsal_ventral_ccf_coordinate': [100.0, 200.0, 300.0]})


def fake_load_lfp(data_root, session_id, channels, start, stop):
    t = np.arange(start, stop, 1 / config.LFP_FS)
    lfp = np.stack([np.sin(2 * np.pi * f * t) + 0.1 * np.random.default_rng(f).normal(size=len(t))
                    for f in (6, 40, 9)], axis=1)
    visp = (channels['structure_acronym'] == 'VISp').to_numpy()
    return channels[visp], lfp[:, visp], t


@pytest.fixture
def pipeline_module(monkeypatch):
    monkeypatch.setitem(sys.modules, 'process_session', types.SimpleNamespace(session=FakeSession))
    monkeypatch.setattr(preprocess_data.pp, 'load_lfp', fake_load_lfp)
    monkeypatch.setattr(config, 'NUM_EPOCHS', 1)
    for name, value in dict(ATTR_SAMPLED=20, ATTR_EVENTS=12, PERI_EVENTS=4, IG_N_STEPS=4, IG_MAX_ERROR=1.0,
                            ATTR_TARGETS=config.ATTR_TARGETS).items():
        monkeypatch.setattr(config, name, value)
    import run_session
    return run_session


@pytest.fixture
def session_dir(tmp_path):
    """A session directory in the AllenSDK cache layout (contents are not used by the fakes)."""
    d = tmp_path / 'visual-behavior-neuropixels-0.4.0' / 'behavior_ecephys_sessions' / str(HVS_SESSION)
    d.mkdir(parents=True)
    return d
