import pandas as pd
import pytest

pytest.importorskip('pynwb')
from process_probe import DATASET_DIR, find_visp_probe


def _write_probes(tmp_path, rows):
    meta = tmp_path / DATASET_DIR / 'project_metadata'
    meta.mkdir(parents=True)
    pd.DataFrame(rows, columns=['ecephys_session_id', 'ecephys_probe_id', 'name', 'structure_acronyms']
                 ).to_csv(meta / 'probes.csv', index=False)


def test_prefers_probe_without_vispm(tmp_path):
    _write_probes(tmp_path, [(1, 10, 'probeA', "['VISpm', 'VISp', 'root']"),
                             (1, 11, 'probeC', "['VISp', 'CA1']"),
                             (2, 20, 'probeB', "['VISp']")])
    assert find_visp_probe(tmp_path, 1) == (11, 'C')


def test_falls_back_to_probe_crossing_vispm(tmp_path):
    # the old rule found no probe here and silently used the last probe of the loop
    _write_probes(tmp_path, [(1, 10, 'probeA', "['LGd', 'root']"),
                             (1, 11, 'probeE', "['VISpm', 'VISp']")])
    assert find_visp_probe(tmp_path, 1) == (11, 'E')


def test_vispm_alone_does_not_count(tmp_path):
    _write_probes(tmp_path, [(1, 10, 'probeA', "['VISpm', 'root']")])
    with pytest.raises(ValueError):
        find_visp_probe(tmp_path, 1)


def _write_nwb(path, probe_id, timestamps, data):
    from datetime import datetime, timezone
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import LFP, ElectricalSeries
    nwb = NWBFile('test', str(probe_id), datetime.now(timezone.utc))
    device = nwb.create_device('probe')
    group = nwb.create_electrode_group('shank', 'test', 'brain', device)
    for ch_id in range(100, 100 + data.shape[1]):
        nwb.add_electrode(id=ch_id, x=0.0, y=0.0, z=0.0, imp=0.0, location='x', filtering='none', group=group)
    region = nwb.create_electrode_table_region(list(range(data.shape[1])), 'all')
    series = ElectricalSeries(name=f'probe_{probe_id}_lfp_data', data=data, electrodes=region, timestamps=timestamps)
    nwb.add_acquisition(LFP(name=f'probe_{probe_id}_lfp', electrical_series=series))
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwb)


def test_load_lfp_reads_window_and_visp_channels(tmp_path):
    import numpy as np
    from process_probe import load_lfp
    _write_probes(tmp_path, [(7, 70, 'probeB', "['VISp', 'VISl']")])
    session_dir = tmp_path / DATASET_DIR / 'behavior_ecephys_sessions' / '7'
    session_dir.mkdir(parents=True)
    ts = np.arange(0, 10, 1 / 1250)
    data = np.tile(np.arange(len(ts))[:, None], (1, 4)).astype(float) * [1, 10, 100, 1000]
    _write_nwb(session_dir / 'probe_probeB_lfp.nwb', 70, ts, data)
    channels = pd.DataFrame({'structure_acronym': ['VISl', 'VISp', 'VISp', 'VISl']}, index=[100, 101, 102, 103])

    chans, lfp, t = load_lfp(tmp_path, 7, channels, 2.0, 3.0)
    assert list(chans.index) == [101, 102]
    assert t[0] == pytest.approx(2.0) and t[-1] == pytest.approx(3.0)      # inclusive window
    np.testing.assert_allclose(lfp[:, 0], data[2500:3751, 1])


def test_load_lfp_rejects_wrong_sampling_rate(tmp_path):
    import numpy as np
    from process_probe import load_lfp
    _write_probes(tmp_path, [(7, 70, 'probeB', "['VISp']")])
    session_dir = tmp_path / DATASET_DIR / 'behavior_ecephys_sessions' / '7'
    session_dir.mkdir(parents=True)
    ts = np.arange(0, 2, 1 / 2500)
    _write_nwb(session_dir / 'probe_probeB_lfp.nwb', 70, ts, np.zeros((len(ts), 2)))
    channels = pd.DataFrame({'structure_acronym': ['VISp', 'VISp']}, index=[100, 101])
    with pytest.raises(ValueError, match='2500'):
        load_lfp(tmp_path, 7, channels, 0.0, 1.0)
