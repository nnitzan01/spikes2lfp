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
NUM_EPOCHS = 15                         # maximum; the epoch with the lowest validation loss is kept
PATIENCE = 5                            # stop after this many epochs without validation improvement
BATCH_SIZE = 32
N_SEEDS = 1                             # independently initialised fits per model (run_session --seeds)
TORCH_SEED = 0                          # seed s uses TORCH_SEED + s (weights and batch order)

# Blocked split of the active epoch (splits.py): train | buffer | validation | buffer | test
TEST_FRACTION = 0.2                     # last 20% of the active epoch
VAL_FRACTION = 0.1                      # the 10% before the test block
BUFFER_CHUNKS = 2                       # 3 s chunks dropped between sets (6 s)

# Legacy random chunk split (preprocess_data.chunk_and_reshape; notebooks only)
TEST_SIZE = 0.2
SPLIT_SEED = 42

# --- Integrated gradients ---------------------------------------------------------------
IG_N_STEPS = 50
IG_N_BATCH = 10                         # sequences per IG batch (x IG_N_STEPS on the GPU at once)
IG_BASELINE = 0                         # zero input = no spikes
IG_MAX_SEQUENCES = 2000                 # SequenceIG: windows x steps sent through the model at once (GPU memory)

# Windows over which attributions are computed (run_session.py, stage 'attr').
#   epoch:    'active', 'spontaneous' or 'session' (start of active -> end of passive)
#   duration: seconds from the start of the epoch (None = whole epoch)
#   targets:  model targets to attribute (None = all; 0 = broadband)
ATTR_WINDOWS = {
    'active':      dict(epoch='active', duration=720, targets=None),       # per-unit summaries
    'spontaneous': dict(epoch='spontaneous', duration=300, targets=None),  # per-unit summaries
    'session':     dict(epoch='session', duration=None, targets=[0]),      # peri-event analyses
}

# --- Peri-event analyses ----------------------------------------------------------------
TOP_K = 40                              # size of the top-K influential set for Jaccard
STIM_WINDOW = (-0.25, 0.5)              # s around image onset
STIM_BASELINE = (-0.25, 0.0)
HVS_WINDOW = (-1.0, 1.0)                # s around the HVS power peak
HVS_BASELINE = (-1.0, -0.75)
