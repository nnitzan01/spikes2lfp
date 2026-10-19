import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

# stimulus_block values in VBN sessions
ACTIVE_BLOCK = 0        # change-detection task (selected via the 'active' column)
SPONTANEOUS_BLOCKS = (1, 3)
GABOR_BLOCK = 2         # receptive-field mapping
FLASH_BLOCK = 4         # full-field flashes
PASSIVE_BLOCK = 5       # passive replay of the task images


class session:
    """
    Units, spike times, channels and stimulus epochs of one Visual Behavior Neuropixels session.

    df: units table (tables/units_info.csv); only units listed in df.unit_id are kept.
    output_dir: AllenSDK S3 cache directory.
    target_areas: optional list of structure acronyms to keep; all areas if None.

    Epoch boundaries are stored as [start, stop] in whole seconds (self.<epoch>_times).
    """
    def __init__(self, session_id, df, output_dir, target_areas=None):
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(output_dir)
        cache.load_manifest("visual-behavior-neuropixels_project_manifest_v0.4.0.json")

        self.id = session_id
        self.session = cache.get_ecephys_session(session_id)
        self.probes = self.session.probes
        self.channels = self.session.get_channels()

        # units with their peak-channel info, restricted to the curated unit list
        units = self.session.get_units().merge(self.channels, left_on='peak_channel_id', right_index=True)
        units = units[units.index.isin(df.unit_id)]
        if target_areas is not None:
            units = units[units.structure_acronym.isin(target_areas)]
        units = units.copy()
        # pool the MG and LG subregions (e.g. MGd/MGv/MGm, LGd/LGv) into one area each
        for nucleus in ('MG', 'LG'):
            units.loc[units['structure_acronym'].str.contains(nucleus), 'structure_acronym'] = nucleus
        self.units = units

        self.spike_times = self.session.spike_times

        stim = self.session.stimulus_presentations
        self.stimulus_presentation = stim
        self.active = stim[stim['active']]
        self.gabor = stim[stim['stimulus_block'] == GABOR_BLOCK]
        self.spontaneous = stim[stim['stimulus_block'].isin(SPONTANEOUS_BLOCKS)]
        self.flash = stim[stim['stimulus_block'] == FLASH_BLOCK]
        self.passive = stim[stim['stimulus_block'] == PASSIVE_BLOCK]

        # task epochs are padded by 1 s on each side
        self.active_times = [int(self.active.start_time.values[0]) - 1, int(self.active.end_time.values[-1]) + 1]
        self.passive_times = [int(self.passive.start_time.values[0]) - 1, int(self.passive.end_time.values[-1]) + 1]
        # only the last spontaneous block (block 3, between RF mapping and flashes) is used
        self.spontaneous_times = [int(self.spontaneous.start_time.values[-1]), int(self.spontaneous.end_time.values[-1])]
        self.flash_times = [int(self.flash.start_time.values[0]), int(self.flash.end_time.values[-1])]
        self.gabor_times = [int(self.gabor.start_time.values[0]), int(self.gabor.end_time.values[-1])]

        self.behavior_id = self.session.behavior_session_id
