"""
Where inputs and outputs live.

Data root = the AllenSDK S3 cache directory. Scripts are given a session directory inside it,
    <data_root>/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>
and write their results under
    <data_root>/spikes2lfp/{models,variables,attrs,plots}/<session_id>/
"""
import os
from pathlib import Path
from types import SimpleNamespace

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_SUBDIR = 'spikes2lfp'

# Data root used by the notebooks; set SPIKES2LFP_DATA_ROOT to override (scripts take the
# session directory on the command line instead).
DATA_ROOT = Path(os.environ.get('SPIKES2LFP_DATA_ROOT',
                                r'Y:\buzsakilab\Buzsakilabspace\LabShare\NoamNitzan\Open_Access\Allen_2022'))


def parse_session_dir(session_dir):
    """Return (data_root, session_id) for a session directory in the layout above."""
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f'{session_dir} is not a directory')
    return session_dir.parent.parent.parent, int(session_dir.name)


def session_output_dirs(data_root, session_id):
    """Output directories of one session (not created here)."""
    base = Path(data_root) / OUTPUT_SUBDIR
    return SimpleNamespace(**{name: base / name / str(session_id)
                              for name in ('models', 'variables', 'attrs', 'plots')})


def model_file(dirs, chan, band):
    return dirs.models / f'lstm_model_chan{chan}_band{band}.pt'


def attribution_file(dirs, window, chan, band):
    """Raw (time bins x units) attributions of one model over one config.ATTR_WINDOWS window, sparse .npz."""
    return dirs.attrs / window / f'attribution_raw_chan{chan}_band{band}.npz'


def hvs_events_file(dirs):
    """HVS events from experiments/detect_cortical_oscillations.py: (events x [start, peak, end]) in seconds."""
    return dirs.variables / 'detected_hvs_events.npy'
