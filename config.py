"""
Settings shared by the spikes -> LFP pipeline, experiments and notebooks.

Change values here rather than in individual scripts. Models saved with one LSTM configuration
cannot be loaded with another, so LSTM_KWARGS must match the models on disk.
"""

# --- Data -------------------------------------------------------------------------------
UNITS_TABLE = 'tables/units_info.csv'   # curated units to include, relative to the repo root
BIN_SIZE = 0.004                        # s; spike-count bins, and the LFP sample period after downsampling
LFP_FS = 1250                           # Hz; LFP sampling rate in the VBN NWB files
LFP_DOWNSAMPLE = 5                      # 1250 Hz / 5 = 250 Hz = 1 / BIN_SIZE

# Model targets: index 0 is the broadband (z-scored) LFP, index i > 0 is the power envelope of BANDS[i-1]
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)]  # Hz
N_TARGETS = len(BANDS) + 1

# --- Model and training -----------------------------------------------------------------
SEQLENGTH = 750                         # bins per sequence (3 s); also the train/test chunk size
LSTM_KWARGS = dict(num_hidden=50, num_layers=1, conv=False)
NUM_EPOCHS = 15
BATCH_SIZE = 32
TEST_SIZE = 0.2                         # fraction of 3 s chunks held out
SPLIT_SEED = 42                         # train/test split
TORCH_SEED = 0                          # weight initialisation and batch shuffling

# --- Integrated gradients ---------------------------------------------------------------
IG_N_STEPS = 50
IG_BASELINE = 0                         # zero input = no spikes
ACTIVE_ATTR_DURATION = 720              # s; window at the start of the active epoch used for the
                                        # per-neuron mean attribution summaries

# --- Peri-event analyses ----------------------------------------------------------------
TOP_K = 40                              # size of the top-K influential set for Jaccard
STIM_WINDOW = (-0.25, 0.5)              # s around image onset
STIM_BASELINE = (-0.25, 0.0)
HVS_WINDOW = (-1.0, 1.0)                # s around the HVS power peak
HVS_BASELINE = (-1.0, -0.75)
