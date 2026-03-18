import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import random_split

from braindecode.classifier import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.util import set_random_seeds

from eegconformer import EEGConformer

# Default experiment config.
# - Band-pass: 4-40 Hz (IIR)
# - Training window: 4 s
# - Embedding dim: 32 (n_filters_time)
# - Temporal kernel: (1, 25), stride (1, 1)
# - Pooling: (1, 95) with stride (1, 11)
# - Self-attention depth: 6, heads: 8
# - Optimizer: Adam, lr = 2e-4, betas = (0.5, 0.999)
# With n_times=1000 (4 s at 250 Hz), this yields sequence length 81.
SEED = 2023
LR = 2e-4
BATCH_SIZE = 64
EPOCHS = 300
N_FILTERS = 32
TEMPORAL_KERNEL = 25
POOL_KERNEL = 95
POOL_STRIDE = 11
ATT_DEPTH = 6
ATT_HEADS = 8
DATASET_NAME = "BNCI2014_001"
ALL_SUBJECT_IDS = tuple(range(1, 10))
ATTENTION_CHOICES = ("multiheadattention", "simpleattention")
N_CLASSES = 4  # four motor imagery classes
DEFAULT_N_SEEDS = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(ALL_SUBJECT_IDS),
        help="Subject ids to run (default: all BNCI2014_001 subjects 1..9).",
    )
    parser.add_argument(
        "--attentions",
        choices=ATTENTION_CHOICES,
        nargs="+",
        default=list(ATTENTION_CHOICES),
        help="Attention mechanisms to evaluate for each subject.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where one JSON result file per subject will be written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size per run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base random seed.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of consecutive seeds to run for each attention mechanism.",
    )
    return parser.parse_args()


def sequence_length_after_patch_embedding(n_times: int) -> int:
    t_after_temporal_conv = n_times - TEMPORAL_KERNEL + 1
    return (t_after_temporal_conv - POOL_KERNEL) // POOL_STRIDE + 1


def prepare_subject_data(subject_id: int, seed: int):
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[subject_id])

    # Paper uses 6th-order Chebyshev 4-40 Hz + z-score. We keep the band-pass and
    # approximate the z-score with exponential moving standardization for stability.
    preprocessors = [
        Preprocessor("pick_types", eeg=True),
        Preprocessor("filter", l_freq=4.0, h_freq=40.0, method="iir"),
        Preprocessor(lambda x: x * 1e6),  # convert to microvolts
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

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
    )

    # Match the paper's session split: session_T for training, session_E for test.
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
            generator=torch.Generator().manual_seed(seed),
        )

    sample, _, _ = train_set[0]
    n_chans, input_window_samples = sample.shape
    return train_set, test_set, n_chans, input_window_samples


def train_and_evaluate(
    train_set,
    test_set,
    n_chans: int,
    input_window_samples: int,
    attention: str,
    epochs: int,
    batch_size: int,
    run_seed: int,
):
    set_random_seeds(seed=run_seed, cuda=torch.cuda.is_available())

    model = EEGConformer(
        n_chans=n_chans,
        n_outputs=N_CLASSES,
        n_times=input_window_samples,
        n_filters_time=N_FILTERS,
        filter_time_length=TEMPORAL_KERNEL,
        pool_time_length=POOL_KERNEL,
        pool_time_stride=POOL_STRIDE,
        num_layers=ATT_DEPTH,
        num_heads=ATT_HEADS,
        attention=attention,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=Adam,
        train_split=None,  # paper trains on session_T, evaluates on session_E
        optimizer__lr=LR,
        optimizer__betas=(0.5, 0.999),
        batch_size=batch_size,
        device=device,
    )

    t0 = time.time()
    clf.fit(train_set, y=None, epochs=epochs)
    duration_sec = time.time() - t0

    y_test = [test_set[i][1] for i in range(len(test_set))]
    test_accuracy = float(clf.score(test_set, y=y_test))

    return {
        "attention": attention,
        "status": "ok",
        "seed": run_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_train_windows": len(train_set),
        "n_test_windows": len(test_set),
        "test_accuracy": test_accuracy,
        "train_duration_sec": duration_sec,
    }


def write_subject_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def main():
    args = parse_args()
    subjects = list(dict.fromkeys(args.subjects))
    attentions = list(dict.fromkeys(args.attentions))
    seeds = [args.seed + i for i in range(args.n_seeds)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subjects:
        print(f"\n=== Subject {subject_id} ===")
        train_set, test_set, n_chans, input_window_samples = prepare_subject_data(
            subject_id, args.seed + subject_id
        )
        sequence_length = sequence_length_after_patch_embedding(input_window_samples)

        subject_result_path = args.output_dir / f"subject_{subject_id:02d}.json"
        subject_payload = {
            "dataset": DATASET_NAME,
            "subject_id": subject_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_chans": n_chans,
                "input_window_samples": input_window_samples,
                "learning_rate": LR,
                "seeds": seeds,
                "n_filters_time": N_FILTERS,
                "filter_time_length": TEMPORAL_KERNEL,
                "pool_time_length": POOL_KERNEL,
                "pool_time_stride": POOL_STRIDE,
                "num_layers": ATT_DEPTH,
                "num_heads": ATT_HEADS,
                "sequence_length": sequence_length,
            },
            "runs": [],
        }

        for attention in attentions:
            for run_seed in seeds:
                print(
                    f"Training subject {subject_id} with attention={attention} "
                    f"(seed={run_seed})..."
                )
                try:
                    run_result = train_and_evaluate(
                        train_set=train_set,
                        test_set=test_set,
                        n_chans=n_chans,
                        input_window_samples=input_window_samples,
                        attention=attention,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        run_seed=run_seed,
                    )
                    subject_payload["runs"].append(run_result)
                    print(
                        f"Subject {subject_id} | attention={attention} | seed={run_seed} | "
                        f"test_accuracy={run_result['test_accuracy']:.6f}"
                    )
                except Exception as exc:
                    subject_payload["runs"].append(
                        {
                            "attention": attention,
                            "status": "failed",
                            "seed": run_seed,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"Subject {subject_id} | attention={attention} | seed={run_seed} failed: {exc}"
                    )
                finally:
                    subject_payload["updated_at_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    write_subject_json(subject_result_path, subject_payload)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print(f"Wrote {subject_result_path}")


if __name__ == "__main__":
    main()
