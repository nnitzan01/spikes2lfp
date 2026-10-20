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
# causal CNN-LSTM: Conv1d over time (16 filters, kernel 3 = the current and 2 previous bins)
# -> LSTM (50 units) -> linear readout at every time step
LSTM_KWARGS = dict(num_hidden=50, num_layers=1, conv=True, kernel_size=3, out_channels=16)
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

# --- History-aware attribution (integrated_gradients.SequenceIG; run_session stage 'attr') ----
# Each target is a bin t; its input window is the ATTR_HISTORY bins ending at t (3 s, the training
# sequence length), and the attributed quantity is the prediction at t.
IG_N_STEPS = 50                         # integration steps (Gauss-Legendre) to start with
IG_MAX_ERROR = 0.01                     # completeness error above which a model is redone with 4x the steps
IG_MAX_STEPS = 400                      # upper limit for those retries
IG_BASELINE = 'zero'                    # 'zero' (no spikes) or 'mean' (each unit's mean training input)
IG_MAX_SEQUENCES = 2000                 # windows x steps sent through the model at once (GPU memory)
IG_BATCH_WINDOWS = 256                  # windows built and attributed per batch
ATTR_HISTORY = SEQLENGTH                # bins of input history per target
ATTR_TARGETS = None                     # model targets to attribute (None = all)
ATTR_SAMPLED = 500                      # random targets drawn from each of: training block, test block, spontaneous epoch
ATTR_EVENTS = 500                       # image presentations whose CSD components are attributed
COMPONENT_TIMES = {'c1': 0.05, 'c2': 0.15, 'c3': 0.30, 'c4': 0.375}   # s after image onset
COMPONENT_TARGET_BINS = 5               # component target = mean prediction over 5 bins (20 ms) ending 8 ms after the time
PERI_EVENTS = 100                       # image presentations with full peri-stimulus attribution maps
PERI_STEP_BINS = 3                      # peri-stimulus targets every 12 ms over STIM_WINDOW
PERI_PER_EVENT_TARGETS = [0]            # model targets whose per-event peri maps are kept (for Jaccard / dominance)
LAG_KERNEL_BIN = 5                      # lag kernels are saved at 20 ms resolution

# --- Peri-event analyses ----------------------------------------------------------------
TOP_K = 40                              # size of the top-K influential set for Jaccard
STIM_WINDOW = (-0.25, 0.5)              # s around image onset
STIM_BASELINE = (-0.25, 0.0)
HVS_WINDOW = (-1.0, 1.0)                # s around the HVS power peak
HVS_BASELINE = (-1.0, -0.75)
