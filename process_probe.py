import pandas as pd
from lfp_utils import *

class probe:
    def __init__(self, session, time_0, time_1):
        assert time_1 > time_0
        self.session_id = session.id
        self.target_area = 'VISp'
        self.session = session
        probe_table = pd.read_csv(r"Y:\buzsakilab\Buzsakilabspace\LabShare\NoamNitzan\Open_Access\Allen_2022\visual-behavior-neuropixels-0.4.0\project_metadata\probes.csv")
        session_probes = probe_table[probe_table['ecephys_session_id'] == self.session_id]
        qualified_probes = []
        for probe_id in session_probes.ecephys_probe_id.values:
            probe = session_probes[session_probes['ecephys_probe_id'] == probe_id]
            if ('VISp' in probe.structure_acronyms.values[0]) and ('VISpm' not in probe.structure_acronyms.values[0]):
                qualified_probes.append(probe_id)
        if len(qualified_probes) == 0:
            print(f'No qualified probes for session {self.session_id}')
        elif len(qualified_probes) > 1:
            print(f'More than one qualified probes for session {self.session_id}')
            print('Choosing the first one')
            self.probe_id = qualified_probes[0]
        else:
            self.probe_id = qualified_probes[0]
        self.probe_letter = probe_table[probe_table['ecephys_probe_id'] == self.probe_id].name.values[0][-1]
        lfp = nwb_to_xarray(load_nwb(self.session_id, self.probe_letter))
        self.all_chans = lfp.channel.values
        self.chans = session.channels.loc[self.all_chans][session.channels.loc[self.all_chans]['structure_acronym'] == 'VISp']
        lfp_sliced = lfp.sel(time=slice(time_0, time_1), channel=self.chans.index.values)
        self.lfp = lfp_sliced.values
        self.bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
        del lfp
