import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

class session:
    def __init__(self, session_id, df, output_dir, target_areas=None):
        
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(output_dir)
        
        self.id = session_id
        self.session = cache.get_ecephys_session(session_id)
        self.probes = self.session.probes
        self.channels = self.session.get_channels()
        
        # add relevent units
        units_raw = self.session.get_units()
        units = units_raw.merge(self.channels, left_on='peak_channel_id', right_index=True)
        units = units[units.index.isin(df.unit_id)]
        # if target_areas are not specified, all units are added
        if target_areas is not None:
            units = units[units.structure_acronym.isin(target_areas)]
        self.units = units
        
        # group together MG and LG subregions
        self.units['structure_acronym'][self.units['structure_acronym'].str.contains('MG')] = 'MG'
        self.units['structure_acronym'][self.units['structure_acronym'].str.contains('LG')] = 'LG'

        self.spike_times = self.session.spike_times
        
        # add stimulus presentation and start/stop times
        # syntax: self.xxx_times -> [start, stop]
        self.stimulus_presentation = self.session.stimulus_presentations
        self.active = self.stimulus_presentation[self.stimulus_presentation['active']]
        self.gabor = self.stimulus_presentation[self.stimulus_presentation['stimulus_block'] == 2]
        self.spontaneous = pd.concat([self.stimulus_presentation[self.stimulus_presentation['stimulus_block'] == 1], self.stimulus_presentation[self.stimulus_presentation['stimulus_block'] == 3]])
        self.flash = self.stimulus_presentation[self.stimulus_presentation['stimulus_block'] == 4]
        self.passive = self.stimulus_presentation[self.stimulus_presentation['stimulus_block'] == 5]
        self.active_times = [int(self.active.start_time.values[0]) - 1, int(self.active.end_time.values[-1]) + 1]
        self.passive_times = [int(self.passive.start_time.values[0]) - 1, int(self.passive.end_time.values[-1]) + 1]
        self.spontaneous_times = [int(self.spontaneous.start_time.values[-1]), int(self.spontaneous.end_time.values[-1])]
        self.flash_times = [int(self.flash.start_time.values[0]), int(self.flash.end_time.values[-1])]
        self.gabor_times = [int(self.gabor.start_time.values[0]), int(self.gabor.end_time.values[-1])]
        # behavior session - not implemented yet
        self.behavior_id = self.session.behavior_session_id
        self.behavior_trials = None # not implemented yet
        self.session_info = None # not implmented yet
