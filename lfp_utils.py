import xarray as xr
import numpy as np
import os
from pynwb import NWBHDF5IO

# base_path = 'Z:/Buzsakilabspace/LabShare/NoamNitzan/Open_Access/Allen_2022/data/'
base_path = 'F:/vbn_s3_cache/visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions'

def load_nwb(session_id, probe_letter):
    probe_letter = probe_letter.upper()
    path = f'{base_path}/session_{session_id}/'
    # session does not exist
    if not os.path.exists(path): 
        print(f'Session {session_id} does not exist')
        return None
    # probe does not exist
    if not os.path.exists(f'{path}/probe{probe_letter}_lfp.nwb'):
        print(f'Probe {probe_letter} does not exist')
        return None
    # load the data
    path = f'{path}/probe{probe_letter}_lfp.nwb'
    io = NWBHDF5IO(path, mode="r") # type: ignore
    nwbfile = io.read()
    return nwbfile

def nwb_to_xarray(nwb):
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
