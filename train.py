import torch
from torch.optim import Adam
from torch.utils.data import random_split

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, exponential_moving_standardize
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.classifier import EEGClassifier
from braindecode.models import EEGConformer
from braindecode.util import set_random_seeds


# Hyperparameters pulled from “EEG Conformer: Convolutional Transformer for EEG
# Decoding and Visualization” (IEEE TNSRE, 2023).
# - Band-pass: 4–40 Hz Chebyshev (approximated here with IIR)
# - Training window: seconds 2–6 of each trial (4 s, non‑overlapping)
# - k (conv channels): 40
# - Temporal kernel: (1, 25), stride (1, 1)
# - Spatial kernel: (ch, 1), stride (1, 1)
# - Pooling: (1, 75) with stride (1, 15)
# - Self‑attention depth: 6, heads: 10
# - Optimizer: Adam, lr = 2e‑4, betas = (0.5, 0.999)
SEED = 2023
LR = 2e-4
BATCH_SIZE = 64
EPOCHS = 300
N_FILTERS = 40
TEMPORAL_KERNEL = 25
POOL_KERNEL = 75
POOL_STRIDE = 15
ATT_DEPTH = 6
ATT_HEADS = 10


set_random_seeds(seed=SEED, cuda=torch.cuda.is_available())

# Dataset: BCI Competition IV 2a (BNCI2014_001), subject 1
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

# Paper uses 6‑order Chebyshev 4–40 Hz + z‑score. We keep the band‑pass and
# approximate the z‑score with exponential moving standardization for stability.
preprocessors = [
    Preprocessor("pick_types", eeg=True),
    Preprocessor("filter", l_freq=4.0, h_freq=40.0, method="iir"),
    Preprocessor(lambda x: x * 1e6),  # convert to µV
    Preprocessor(
        exponential_moving_standardize,
        factor_new=1e-3,
        init_block_size=1000,
        eps=1e-4,
    ),
]
preprocess(dataset, preprocessors)

sfreq = dataset.datasets[0].raw.info["sfreq"]
window_size_samples = int(4 * sfreq)

trial_start_offset_samples = 0
trial_stop_offset_samples = 0

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=trial_stop_offset_samples,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    preload=True,
)

# Match the paper’s session split: session_T for training, session_E for test.
splits = windows_dataset.split("session")
if "session_T" in splits and "session_E" in splits:
    train_set = splits["session_T"]
    test_set = splits["session_E"]
else:
    # Fallback to an 80/20 random split if session labels are unavailable.
    n_train = int(len(windows_dataset) * 0.8)
    train_set, test_set = random_split(
        windows_dataset,
        [n_train, len(windows_dataset) - n_train],
        generator=torch.Generator().manual_seed(SEED),
    )

# Infer shape from a single window.
sample, _, _ = train_set[0]
n_chans, input_window_samples = sample.shape
n_classes = 4  # four motor imagery classes

model = EEGConformer(
    n_chans=n_chans,
    n_outputs=n_classes,
    n_times=input_window_samples,
    n_filters_time=N_FILTERS,
    filter_time_length=TEMPORAL_KERNEL,
    pool_time_length=POOL_KERNEL,
    pool_time_stride=POOL_STRIDE,
    num_layers=ATT_DEPTH,
    num_heads=ATT_HEADS,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=Adam,
    train_split=None,  # paper trains on session_T, evaluates on session_E
    optimizer__lr=LR,
    optimizer__betas=(0.5, 0.999),
    batch_size=BATCH_SIZE,
    device=device,
)

clf.fit(train_set, y=None, epochs=EPOCHS)

y_test = [test_set[i][1] for i in range(len(test_set))]
acc = clf.score(test_set, y=y_test)
print("Test accuracy:", acc)
