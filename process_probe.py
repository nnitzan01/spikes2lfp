import os
import numpy as np
import xarray as xr
import pandas as pd
from pynwb import NWBHDF5IO
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_nwb(nwb_dir, session_id, probe_letter):
    """
    Instead of downloading LFP data using AllenSDK, LFP is loaded directly from .nwb files.
    Faster this way.

    input:
    session_id: int, session_id
    probe_letter: str, probe letter, e.g. 'A', 'B', 'C', etc.
    
    output:
    nwbfile: NWBFile object
    
    note:
    assuming the following file structure:
    
    nwb_dir/session_id/probeA_lfp.nwb
    """
    probe_letter = probe_letter.upper()
    session_path = Path(Path(nwb_dir) /"visual-behavior-neuropixels-0.4.0"/"behavior_ecephys_sessions"/str(session_id))
    # session folder does not exist
    if not os.path.exists(session_path): 
        print(f'Session {session_id} does not exist')
        return None
    # probe does not exist
    probe_nwb_path = Path(session_path / f'probe_probe{probe_letter}_lfp.nwb')
    if not os.path.exists(probe_nwb_path):
        print(f'Probe {probe_letter} does not exist')
        return None
    # load the data
    io = NWBHDF5IO(probe_nwb_path, mode="r") # type: ignore
    nwbfile = io.read()
    return nwbfile

def nwb_to_xarray(nwb):
    """
    Reading from the provided .nwb file and converting it to xarray DataArray.

    input:
    nwb: NWBFile object

    output:
    lfp: xarray DataArray, LFP data
        attributes: timestamps - time points for each LFP recording
                    channels - channel ids for each channel 
    """
    if nwb is None:
        return None
    else:
        probe_id = nwb.identifier
        data = nwb.acquisition[f'probe_{probe_id}_lfp'].electrical_series[f'probe_{probe_id}_lfp_data']
        lfp = np.array(data.data)/2
        timestamps = np.array(data.timestamps)
        electrodes = nwb.electrodes.to_dataframe()
        channels = electrodes.index.values
        return xr.DataArray(lfp, coords=[timestamps, channels], dims=['time', 'channel']) # type: ignore


def load_lfp(output_dir, id, channels, start_time, stop_time):
    """
    Reading from .nwb files directly and loading LFP data for the given session.

    input:
    output_dir: str, path to the output directory
    session: object, AllenSDK session object

    output:
    chans: DataFrame, VISp channels
    lfp_sliced: xarray DataArray, LFP data for the qualified channels
    """
    probe_table_dir = Path(output_dir / "visual-behavior-neuropixels-0.4.0" / "project_metadata" / "probes.csv")
    probe_table = pd.read_csv(probe_table_dir)
    session_probes = probe_table[probe_table['ecephys_session_id'] == id]
    qualified_probes = []
    for probe_id in session_probes.ecephys_probe_id.values:
        probe = session_probes[session_probes['ecephys_probe_id'] == probe_id]
        if ('VISp' in probe.structure_acronyms.values[0]) and ('VISpm' not in probe.structure_acronyms.values[0]):
            qualified_probes.append(probe_id)
    if len(qualified_probes) == 0:
        print(f'No qualified probes for session {id}')
    elif len(qualified_probes) > 1:
        print(f'More than one qualified probes for session {id}')
        print('Choosing the first one')
        probe_id = qualified_probes[0]
    else:
        probe_id = qualified_probes[0]
    probe_letter = probe_table[probe_table['ecephys_probe_id'] == probe_id].name.values[0][-1]
    lfp = nwb_to_xarray(load_nwb(output_dir, id, probe_letter))
    all_chans = lfp.channel.values
    chans = channels.loc[all_chans][channels.loc[all_chans]['structure_acronym'] == 'VISp']
    lfp_sliced = lfp.sel(time=slice(start_time, stop_time), channel=chans.index.values)
    return chans, lfp_sliced.data, lfp_sliced.time.to_numpy()