"""
Load VISp LFP for a session directly from the Visual Behavior Neuropixels NWB files.

Reading the probe NWB files with pynwb is much faster than going through the AllenSDK cache.
Expected layout under the data root (the AllenSDK S3 cache directory):

    visual-behavior-neuropixels-0.4.0/project_metadata/probes.csv
    visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>/probe_probe<X>_lfp.nwb
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

import config

DATASET_DIR = 'visual-behavior-neuropixels-0.4.0'


def find_visp_probe(data_root, session_id):
    """
    Return (probe_id, probe_letter) of the probe that recorded VISp in this session.

    Probes that do not also pass through VISpm are preferred (this was the original selection
    rule, kept so existing sessions use the same probe); a probe crossing both is used only if
    there is no other. Among equals, the first in probes.csv wins.
    """
    probe_table = pd.read_csv(Path(data_root) / DATASET_DIR / 'project_metadata' / 'probes.csv')
    session_probes = probe_table[probe_table['ecephys_session_id'] == session_id]
    # structure_acronyms is stored as the string repr of a list, e.g. "['VISp', 'VISpm', 'root']";
    # parse it so 'VISp' is matched exactly rather than as a substring of 'VISpm'
    areas = session_probes['structure_acronyms'].apply(ast.literal_eval)
    qualified = session_probes[areas.apply(lambda a: 'VISp' in a)]
    if qualified.empty:
        raise ValueError(f'No probe in session {session_id} recorded VISp')
    also_visp_m = areas[qualified.index].apply(lambda a: 'VISpm' in a)
    qualified = pd.concat([qualified[~also_visp_m], qualified[also_visp_m]])
    if len(qualified) > 1:
        print(f'{len(qualified)} probes recorded VISp in session {session_id}; using {qualified.iloc[0]["name"]}')
    probe = qualified.iloc[0]
    return probe['ecephys_probe_id'], probe['name'][-1]   # name is e.g. 'probeC'


def load_lfp(data_root, session_id, channels, start_time, stop_time):
    """
    Load the VISp LFP of one session between start_time and stop_time (s).

    input:
    data_root: str or Path, AllenSDK cache directory (see module docstring)
    session_id: int
    channels: DataFrame, session channel table (session.get_channels()), indexed by channel id
    start_time, stop_time: float, seconds on the session clock

    output:
    chans: DataFrame, rows of `channels` for the VISp channels, in recording order
    lfp: ndarray (time x channels)
    timestamps: ndarray (time,)
    """
    _, probe_letter = find_visp_probe(data_root, session_id)
    nwb_path = (Path(data_root) / DATASET_DIR / 'behavior_ecephys_sessions' / str(session_id)
                / f'probe_probe{probe_letter.upper()}_lfp.nwb')
    if not nwb_path.exists():
        raise FileNotFoundError(nwb_path)

    with NWBHDF5IO(str(nwb_path), mode='r') as io:
        nwb = io.read()
        probe_id = nwb.identifier
        series = nwb.acquisition[f'probe_{probe_id}_lfp'].electrical_series[f'probe_{probe_id}_lfp_data']
        all_timestamps = np.asarray(series.timestamps)
        # read only the requested time window from disk instead of the whole recording
        i0 = np.searchsorted(all_timestamps, start_time, side='left')
        i1 = np.searchsorted(all_timestamps, stop_time, side='right')   # inclusive, like the old xarray slice
        lfp = np.asarray(series.data[i0:i1])
        channel_ids = nwb.electrodes.to_dataframe().index.values

    # The NWB metadata states a 2500 Hz rate although the LFP is sampled at 1250 Hz. Timing here
    # comes from the stored timestamps, so check the real rate instead of trusting the metadata.
    fs = 1 / np.median(np.diff(all_timestamps[i0:i1]))
    if abs(fs - config.LFP_FS) > 1:
        raise ValueError(f'LFP sampled at {fs:.1f} Hz, expected {config.LFP_FS} Hz ({nwb_path})')

    probe_chans = channels.loc[channel_ids]
    is_visp = (probe_chans['structure_acronym'] == 'VISp').to_numpy()
    return probe_chans[is_visp], lfp[:, is_visp], all_timestamps[i0:i1]
