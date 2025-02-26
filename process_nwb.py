import os
from pynwb import NWBHDF5IO

class session:
    def __init__(self, session_id):
        self.id = session_id
        # remember to change this path
        self.base_path = 'E:/vbn_s3_cache/visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions'
        filepath = f'{self.base_path}/{self.id}/ecephys_session_{self.id}.nwb'
        if not os.path.exists(filepath): print(f'Session {self.id} does not exist')
        io = NWBHDF5IO(filepath, mode="r")
        self.nwb_file = io.read()
        self.probes = None
        self.units = None
        self.spikes = None
        self.stimulus_presentation = None
        self.channels = None
        self.behavior_id = None
        self.behavior_trials = None
        self.session_info = None
        self.start = None
        self.stop = None

    def extract(self):
        # things to load from nwb
        # 1. units
        self.units = self.nwb_file.units.to_dataframe()
        # 2. spikes -> store in the form of dictionaries
        # 3. stimulus presentation
        imgs = self.nwb_file.intervals


        # 4. probes
        probes_list = list(self.nwb_file.electrode_groups.keys())
        

        # 5. channels
        

        # 6. bahevior id
        self.behavior_id = self.nwb_file.lab_meta_data['metadata'].behavior_session_id
        
        # 7. behavior session
        ### not implemented yet ###

        # 8. add the session info sheet
        ### not implemented yet ###

        # 9. start
        self.start = self.nwb_file.trials.to_dataframe().start_time.values[0]

        # 10. stop
        self.stop = self.nwb_file.trials.to_dataframe().stop_time.values[-1]
    
    