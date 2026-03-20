import argparse
import json
import math
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

DATASET_NAME = "BNCI2014_001"
ALL_SUBJECT_IDS = tuple(range(1, 10))
ATTENTION_CHOICES = ("multiheadattention", "simpleattention")
N_CLASSES = 4

BASE_SEED = 2023
DEFAULT_N_SEEDS = 5

# Best simple-attention candidate found from:
# results/run_20260319_225300, candidate_id=6
BEST_CONFIG = {
    "learning_rate": 2e-4,
    "batch_size": 64,
    "epochs": 300,
    "n_filters_time": 32,
    "filter_time_length": 25,
    "pool_time_length": 95,
    "pool_time_stride": 11,
    "drop_prob": 0.5,
    "num_layers": 4,
    "num_heads": 4,
    "att_drop_prob": 0.3,
    "optimizer": "Adam",
    "optimizer_betas": (0.5, 0.999),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train EEGConformer with both attentions using the best tuned config "
            "from simpleattention search."
        )
    )
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
        help="Attention mechanisms to compare using the same best config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where one JSON file per subject is written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help="Base random seed.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of consecutive seeds to run per attention.",
    )
    parser.add_argument(
        "--best-run-dir",
        type=Path,
        default=Path("results/run_20260319_225300"),
        help="Run folder where the best simple-attention config came from.",
    )
    parser.add_argument(
        "--best-candidate-id",
        type=int,
        default=6,
        help="Candidate id used as the selected best simple-attention config.",
    )
    return parser.parse_args()


def sequence_length_after_patch_embedding(
    n_times: int,
    filter_time_length: int,
    pool_time_length: int,
    pool_time_stride: int,
) -> int:
    t_after_temporal_conv = n_times - filter_time_length + 1
    sequence_length = (t_after_temporal_conv - pool_time_length) // pool_time_stride + 1
    if sequence_length <= 0:
        raise ValueError(
            "Invalid patch-embedding settings produced non-positive sequence length: "
            f"n_times={n_times}, filter_time_length={filter_time_length}, "
            f"pool_time_length={pool_time_length}, pool_time_stride={pool_time_stride}"
        )
    return sequence_length


def prepare_subject_data(subject_id: int, seed: int):
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[subject_id])

    preprocessors = [
        Preprocessor("pick_types", eeg=True),
        Preprocessor("filter", l_freq=4.0, h_freq=40.0, method="iir"),
        Preprocessor(lambda x: x * 1e6),
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

    splits = windows_dataset.split("session")
    if "session_T" in splits and "session_E" in splits:
        train_set = splits["session_T"]
        test_set = splits["session_E"]
    else:
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
    run_seed: int,
    config: dict,
) -> dict:
    set_random_seeds(seed=run_seed, cuda=torch.cuda.is_available())

    model = EEGConformer(
        n_chans=n_chans,
        n_outputs=N_CLASSES,
        n_times=input_window_samples,
        n_filters_time=config["n_filters_time"],
        filter_time_length=config["filter_time_length"],
        pool_time_length=config["pool_time_length"],
        pool_time_stride=config["pool_time_stride"],
        drop_prob=config["drop_prob"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        att_drop_prob=config["att_drop_prob"],
        attention=attention,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=Adam,
        train_split=None,
        optimizer__lr=config["learning_rate"],
        optimizer__betas=tuple(config["optimizer_betas"]),
        batch_size=config["batch_size"],
        device=device,
    )

    t0 = time.time()
    clf.fit(train_set, y=None, epochs=config["epochs"])
    duration_sec = time.time() - t0

    y_test = [test_set[i][1] for i in range(len(test_set))]
    test_accuracy = float(clf.score(test_set, y=y_test))

    return {
        "status": "ok",
        "test_accuracy": test_accuracy,
        "train_duration_sec": duration_sec,
        "n_train_windows": len(train_set),
        "n_test_windows": len(test_set),
    }


def summarize_attention(runs: list[dict], attention: str) -> dict:
    subset = [
        run
        for run in runs
        if run.get("attention") == attention and run.get("status") == "ok"
    ]
    failed_count = len(
        [
            run
            for run in runs
            if run.get("attention") == attention and run.get("status") != "ok"
        ]
    )

    if not subset:
        return {
            "attention": attention,
            "n_ok_runs": 0,
            "n_failed_runs": failed_count,
            "mean_test_accuracy": None,
            "std_test_accuracy": None,
            "min_test_accuracy": None,
            "max_test_accuracy": None,
        }

    scores = [run["test_accuracy"] for run in subset]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)

    return {
        "attention": attention,
        "n_ok_runs": len(subset),
        "n_failed_runs": failed_count,
        "mean_test_accuracy": mean,
        "std_test_accuracy": math.sqrt(variance),
        "min_test_accuracy": min(scores),
        "max_test_accuracy": max(scores),
    }


def write_subject_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def main() -> None:
    args = parse_args()
    subjects = list(dict.fromkeys(args.subjects))
    attentions = list(dict.fromkeys(args.attentions))
    seeds = [args.seed + i for i in range(args.n_seeds)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "learning_rate": float(BEST_CONFIG["learning_rate"]),
        "batch_size": int(BEST_CONFIG["batch_size"]),
        "epochs": int(BEST_CONFIG["epochs"]),
        "n_filters_time": int(BEST_CONFIG["n_filters_time"]),
        "filter_time_length": int(BEST_CONFIG["filter_time_length"]),
        "pool_time_length": int(BEST_CONFIG["pool_time_length"]),
        "pool_time_stride": int(BEST_CONFIG["pool_time_stride"]),
        "drop_prob": float(BEST_CONFIG["drop_prob"]),
        "num_layers": int(BEST_CONFIG["num_layers"]),
        "num_heads": int(BEST_CONFIG["num_heads"]),
        "att_drop_prob": float(BEST_CONFIG["att_drop_prob"]),
        "optimizer": str(BEST_CONFIG["optimizer"]),
        "optimizer_betas": [
            float(BEST_CONFIG["optimizer_betas"][0]),
            float(BEST_CONFIG["optimizer_betas"][1]),
        ],
    }

    print(
        "Running best-config comparison with "
        f"attentions={attentions}, seeds={seeds}, config={config}"
    )

    for subject_id in subjects:
        print(f"\n=== Subject {subject_id} ===")
        train_set, test_set, n_chans, input_window_samples = prepare_subject_data(
            subject_id, args.seed + subject_id
        )

        config_with_shape = dict(config)
        config_with_shape["sequence_length"] = sequence_length_after_patch_embedding(
            n_times=input_window_samples,
            filter_time_length=config["filter_time_length"],
            pool_time_length=config["pool_time_length"],
            pool_time_stride=config["pool_time_stride"],
        )

        subject_result_path = args.output_dir / f"subject_{subject_id:02d}.json"
        subject_payload = {
            "dataset": DATASET_NAME,
            "subject_id": subject_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_type": "best_config_attention_comparison",
            "best_config_source": {
                "run_dir": str(args.best_run_dir),
                "candidate_id": int(args.best_candidate_id),
                "derived_from_attention": "simpleattention",
            },
            "seed_policy": {
                "base_seed": args.seed,
                "n_seeds": args.n_seeds,
                "seeds": seeds,
            },
            "subject_data": {
                "n_chans": n_chans,
                "input_window_samples": input_window_samples,
            },
            "attentions": attentions,
            "config": config_with_shape,
            "runs": [],
        }

        for attention in attentions:
            for run_seed in seeds:
                print(
                    f"Training subject {subject_id} | attention={attention} | seed={run_seed}..."
                )
                try:
                    metrics = train_and_evaluate(
                        train_set=train_set,
                        test_set=test_set,
                        n_chans=n_chans,
                        input_window_samples=input_window_samples,
                        attention=attention,
                        run_seed=run_seed,
                        config=config,
                    )
                    run_result = {
                        "status": "ok",
                        "attention": attention,
                        "seed": run_seed,
                        "hyperparameters": {
                            **config_with_shape,
                            "attention": attention,
                        },
                        **metrics,
                    }
                    subject_payload["runs"].append(run_result)
                    print(
                        f"Subject {subject_id} | attention={attention} | seed={run_seed} | "
                        f"test_accuracy={run_result['test_accuracy']:.6f}"
                    )
                except Exception as exc:
                    subject_payload["runs"].append(
                        {
                            "status": "failed",
                            "attention": attention,
                            "seed": run_seed,
                            "hyperparameters": {
                                **config_with_shape,
                                "attention": attention,
                            },
                            "error": str(exc),
                        }
                    )
                    print(
                        f"Subject {subject_id} | attention={attention} | seed={run_seed} failed: {exc}"
                    )
                finally:
                    subject_payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                    write_subject_json(subject_result_path, subject_payload)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        summaries = [
            summarize_attention(subject_payload["runs"], attention)
            for attention in attentions
        ]
        summaries.sort(
            key=lambda item: (
                item["mean_test_accuracy"] is not None,
                item["mean_test_accuracy"] if item["mean_test_accuracy"] is not None else -1.0,
            ),
            reverse=True,
        )
        subject_payload["attention_summaries"] = summaries
        subject_payload["best_attention"] = summaries[0] if summaries else None
        subject_payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_subject_json(subject_result_path, subject_payload)
        print(f"Wrote {subject_result_path}")


if __name__ == "__main__":
    main()
