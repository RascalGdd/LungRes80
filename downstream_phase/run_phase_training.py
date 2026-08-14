import argparse
import copy
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import random
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.transforms.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from datasets.transforms.optim_factory import (
    create_optimizer,
    get_parameter_groups,
    LayerDecayValueAssigner,
)

from downstream_phase.datasets_phase import build_dataset
from downstream_phase.engine_for_phase import (
    collect_prediction_history_records,
    train_one_epoch,
    validation_one_epoch,
    final_phase_test,
    merge,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
import utils

from model.visual_backbone import visual_backbone
from model.lungreco import lungreco


class VideoSequentialDistributedBatchSampler:
    """Shard whole videos and advance independent chronological stream slots.

    A batch can contain concurrent frames from different videos, but never two
    frames from the same video. When a video finishes, its slot continues with
    another whole video. This preserves autoregressive predicted histories and
    avoids the many tiny batches produced by a global round-robin sampler.
    """

    def __init__(self, dataset, batch_size, num_replicas=1, rank=0):
        self.batch_size = int(batch_size)
        by_video = OrderedDict()
        for index, sample in enumerate(dataset.dataset_samples):
            by_video.setdefault(str(sample["video_id"]), []).append(index)

        assignments = [[] for _ in range(num_replicas)]
        loads = [0 for _ in range(num_replicas)]
        for video_id, indices in sorted(
            by_video.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            target_rank = min(range(num_replicas), key=lambda value: loads[value])
            assignments[target_rank].append((video_id, indices))
            loads[target_rank] += len(indices)
        rank_videos = assignments[rank]
        self.slots = [[] for _ in range(self.batch_size)]
        slot_loads = [0 for _ in range(self.batch_size)]
        for video_id, indices in sorted(
            rank_videos, key=lambda item: (-len(item[1]), item[0])
        ):
            target_slot = min(
                range(self.batch_size), key=lambda value: slot_loads[value]
            )
            self.slots[target_slot].append((video_id, indices))
            slot_loads[target_slot] += len(indices)
        self.slot_loads = slot_loads

    def __iter__(self):
        video_positions = [0] * len(self.slots)
        frame_positions = [0] * len(self.slots)
        while True:
            batch = []
            for slot_index, videos in enumerate(self.slots):
                while video_positions[slot_index] < len(videos):
                    _, indices = videos[video_positions[slot_index]]
                    if frame_positions[slot_index] < len(indices):
                        break
                    video_positions[slot_index] += 1
                    frame_positions[slot_index] = 0
                if video_positions[slot_index] >= len(videos):
                    continue
                _, indices = videos[video_positions[slot_index]]
                batch.append(indices[frame_positions[slot_index]])
                frame_positions[slot_index] += 1
            if batch:
                yield batch
            else:
                break

    def __len__(self):
        return max(self.slot_loads, default=0)


def _latest_prediction_history_bank(output_dir):
    paths = sorted((Path(output_dir) / "history_banks").glob("history_bank_epoch_*.pt"))
    return paths[-1] if paths else None


def _apply_prediction_history_bank(dataset, bank):
    # Sparse target sampling uses Subset so the underlying per-video frame
    # table remains complete for causal clip loading.  Histories are attached
    # to that complete table and are then visible through the subset indices.
    history_dataset = dataset
    while isinstance(history_dataset, torch.utils.data.Subset):
        history_dataset = history_dataset.dataset
    if not hasattr(history_dataset, "dataset_samples"):
        raise TypeError("Prediction history requires a frame-backed dataset")
    dataset_samples = history_dataset.dataset_samples
    histories = bank["histories"]
    counts = bank["counts"]
    history_times = bank.get("history_times")
    bank_video_ids = bank.get("video_ids")
    bank_frame_ids = bank.get("frame_ids")
    if bank_video_ids is not None and bank_frame_ids is not None:
        lookup = {
            (str(video_id).zfill(2), int(frame_id)): index
            for index, (video_id, frame_id) in enumerate(
                zip(bank_video_ids, bank_frame_ids)
            )
        }
        bank_indices = []
        missing = []
        for sample in dataset_samples:
            key = (
                str(sample["video_id"]).zfill(2),
                int(sample["frame_id"]),
            )
            if key not in lookup:
                missing.append(key)
            else:
                bank_indices.append(lookup[key])
        if missing:
            raise ValueError(
                f"Static history bank misses {len(missing)} training frames; "
                f"first missing key={missing[0]}"
            )
    else:
        if len(histories) != len(dataset_samples):
            raise ValueError("Prediction history bank does not match the training set")
        bank_indices = list(range(len(dataset_samples)))
    for sample, bank_index in zip(dataset_samples, bank_indices):
        count = int(counts[bank_index])
        sample["prediction_recent_phases"] = histories[
            bank_index, :count
        ].tolist()
        if history_times is not None:
            sample["prediction_recent_phase_times"] = history_times[
                bank_index, :count
            ].tolist()


def _causal_distinct_phase_histories(
    predictions, max_phases=3, excluded_phase_ids=(6,)
):
    """Return histories from strictly earlier distinct predictions only."""
    excluded = {int(value) for value in excluded_phase_ids}
    recent = []
    histories = []
    for prediction in predictions:
        histories.append(list(recent))
        prediction = int(prediction)
        if prediction in excluded:
            continue
        if prediction in recent:
            recent.remove(prediction)
        recent.append(prediction)
        recent = recent[-int(max_phases) :]
    return histories


def _causal_distinct_phase_histories_with_times(
    predictions, frame_ids, max_phases=3, excluded_phase_ids=(6,), fps=1.0
):
    """Return strictly earlier distinct phases and their transition times."""
    excluded = {int(value) for value in excluded_phase_ids}
    recent = []
    histories = []
    last_prediction = None
    for prediction, frame_id in zip(predictions, frame_ids):
        histories.append(list(recent))
        prediction = int(prediction)
        if prediction != last_prediction and prediction not in excluded:
            recent = [entry for entry in recent if entry[0] != prediction]
            recent.append((prediction, int(round(float(frame_id) / float(fps)))))
            recent = recent[-int(max_phases) :]
        last_prediction = prediction
    return histories


def _build_gt_history_bank(
    dataset,
    max_phases=3,
    excluded_phase_ids=(6,),
    fps=1.0,
    frame_history=False,
):
    """Build strictly causal transition histories from clean training GT labels."""
    history_dataset = dataset
    while isinstance(history_dataset, torch.utils.data.Subset):
        history_dataset = history_dataset.dataset
    if not hasattr(history_dataset, "dataset_samples"):
        raise TypeError("GT history requires a frame-backed dataset")
    samples = history_dataset.dataset_samples
    by_video = OrderedDict()
    for index, sample in enumerate(samples):
        by_video.setdefault(str(sample["video_id"]).zfill(2), []).append(index)

    histories = np.full((len(samples), int(max_phases)), -1, dtype=np.int8)
    history_times = np.full((len(samples), int(max_phases)), -1, dtype=np.int32)
    counts = np.zeros(len(samples), dtype=np.int8)
    for indices in by_video.values():
        indices.sort(key=lambda value: int(samples[value]["frame_id"]))
        if frame_history:
            recent = []
            video_histories = []
            for index in indices:
                video_histories.append(list(recent))
                recent.append(
                    (
                        int(samples[index]["phase_gt"]),
                        int(round(float(samples[index]["frame_id"]) / float(fps))),
                    )
                )
                recent = recent[-int(max_phases):]
        else:
            video_histories = _causal_distinct_phase_histories_with_times(
                [int(samples[index]["phase_gt"]) for index in indices],
                [int(samples[index]["frame_id"]) for index in indices],
                max_phases=max_phases,
                excluded_phase_ids=excluded_phase_ids,
                fps=fps,
            )
        for index, history in zip(indices, video_histories):
            counts[index] = len(history)
            if history:
                histories[index, : len(history)] = [entry[0] for entry in history]
                history_times[index, : len(history)] = [entry[1] for entry in history]

    return {
        "source_epoch": -1,
        "source": "strictly_causal_ground_truth",
        "histories": torch.from_numpy(histories),
        "history_times": torch.from_numpy(history_times),
        "counts": torch.from_numpy(counts),
        "video_ids": [str(sample["video_id"]).zfill(2) for sample in samples],
        "frame_ids": [int(sample["frame_id"]) for sample in samples],
    }


def _apply_prediction_json_histories(
    dataset,
    prediction_path,
    max_phases=3,
    excluded_phase_ids=(6,),
    prediction_field="prediction",
):
    """Attach frozen causal histories from a saved prediction file."""
    rows_by_video = OrderedDict()
    with open(prediction_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_video.setdefault(str(row["video_id"]), []).append(row)
    histories_by_key = {}
    for video_id, rows in rows_by_video.items():
        rows.sort(key=lambda row: int(row["frame_id"]))
        histories = _causal_distinct_phase_histories_with_times(
            [row[prediction_field] for row in rows],
            [row["frame_id"] for row in rows],
            max_phases=max_phases,
            excluded_phase_ids=excluded_phase_ids,
            fps=1.0,
        )
        for row, history in zip(rows, histories):
            histories_by_key[(video_id, int(row["frame_id"]))] = history
    missing = []
    for sample in dataset.dataset_samples:
        key = (str(sample["video_id"]), int(sample["frame_id"]))
        if key not in histories_by_key:
            missing.append(key)
            continue
        history = histories_by_key[key]
        sample["prediction_recent_phases"] = [entry[0] for entry in history]
        sample["prediction_recent_phase_times"] = [entry[1] for entry in history]
    if missing:
        raise ValueError(
            f"Saved predictions do not cover {len(missing)} dataset frames; "
            f"first missing key={missing[0]}"
        )
    if len(histories_by_key) != len(dataset.dataset_samples):
        raise ValueError(
            "Saved prediction count does not exactly match the target dataset"
        )
    print(
        f"Loaded frozen causal histories for {len(histories_by_key)} frames "
        f"from {prediction_path}"
    )


def _build_prediction_history_bank(
    dataset,
    records,
    output_dir,
    epoch,
    num_tasks,
    gaussian_sigma,
    gaussian_history_length,
    transition_history_length,
    distinct_non_other=False,
):
    """Merge random-order predictions into causal per-video transition histories."""
    bank_dir = Path(output_dir) / "history_banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    rank = utils.get_rank()
    probabilities = (
        np.concatenate(records["probabilities"], axis=0)
        if records["probabilities"]
        else np.empty((0, 7), dtype=np.float32)
    )
    torch.save(
        {
            "indices": torch.tensor(records["indices"], dtype=torch.long),
            "probabilities": torch.from_numpy(probabilities).to(torch.float16),
        },
        bank_dir / f"epoch_{epoch:03d}_rank_{rank}.pt",
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    bank_path = bank_dir / f"history_bank_epoch_{epoch:03d}.pt"
    if utils.is_main_process():
        previous_path = _latest_prediction_history_bank(output_dir)
        if previous_path is not None:
            previous = torch.load(previous_path, map_location="cpu")
            all_probabilities = previous["probabilities"].float().numpy()
        else:
            all_probabilities = np.full(
                (len(dataset.dataset_samples), 7), 1.0 / 7.0, dtype=np.float32
            )
        updated = np.zeros(len(dataset.dataset_samples), dtype=bool)
        for task in range(num_tasks):
            shard = torch.load(
                bank_dir / f"epoch_{epoch:03d}_rank_{task}.pt",
                map_location="cpu",
            )
            indices = shard["indices"].numpy()
            all_probabilities[indices] = shard["probabilities"].float().numpy()
            updated[indices] = True

        by_video = OrderedDict()
        for index, sample in enumerate(dataset.dataset_samples):
            by_video.setdefault(str(sample["video_id"]), []).append(index)
        max_phases = (
            int(transition_history_length)
            if distinct_non_other
            else int(transition_history_length) + 1
        )
        histories = np.full(
            (len(dataset.dataset_samples), max_phases), -1, dtype=np.int8
        )
        history_times = np.full(
            (len(dataset.dataset_samples), max_phases), -1, dtype=np.int32
        )
        counts = np.zeros(len(dataset.dataset_samples), dtype=np.int8)
        ages = np.arange(int(gaussian_history_length), dtype=np.float32)
        kernel = np.exp(-0.5 * (ages / float(gaussian_sigma)) ** 2)
        for indices in by_video.values():
            indices.sort(
                key=lambda value: int(dataset.dataset_samples[value]["frame_id"])
            )
            values = all_probabilities[indices]
            denominator = np.convolve(
                np.ones(len(indices), dtype=np.float32), kernel, mode="full"
            )[: len(indices)]
            smoothed = np.empty_like(values)
            for label in range(values.shape[1]):
                smoothed[:, label] = np.convolve(
                    values[:, label], kernel, mode="full"
                )[: len(indices)] / denominator
            predictions = smoothed.argmax(axis=1)
            if distinct_non_other:
                video_histories = _causal_distinct_phase_histories_with_times(
                    predictions,
                    [dataset.dataset_samples[index]["frame_id"] for index in indices],
                    max_phases=max_phases,
                    excluded_phase_ids=(6,),
                    fps=1.0,
                )
            else:
                recent = []
                video_histories = []
                last_prediction = None
                for prediction, index in zip(predictions, indices):
                    video_histories.append(list(recent))
                    prediction = int(prediction)
                    if prediction != last_prediction:
                        recent.append(
                            (
                                prediction,
                                int(dataset.dataset_samples[index]["frame_id"]),
                            )
                        )
                        recent = recent[-max_phases:]
                    last_prediction = prediction
            for index, history in zip(indices, video_histories):
                counts[index] = len(history)
                if history:
                    histories[index, : len(history)] = [entry[0] for entry in history]
                    history_times[index, : len(history)] = [
                        entry[1] for entry in history
                    ]
        torch.save(
            {
                "source_epoch": int(epoch),
                "probabilities": torch.from_numpy(all_probabilities).to(torch.float16),
                "histories": torch.from_numpy(histories),
                "history_times": torch.from_numpy(history_times),
                "counts": torch.from_numpy(counts),
                "video_ids": [
                    str(sample["video_id"]).zfill(2)
                    for sample in dataset.dataset_samples
                ],
                "frame_ids": [
                    int(sample["frame_id"]) for sample in dataset.dataset_samples
                ],
                "updated_samples": int(updated.sum()),
            },
            bank_path,
        )
        print(
            f"Prediction history bank epoch {epoch}: "
            f"updated={int(updated.sum())}/{len(updated)}, path={bank_path}"
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    bank = torch.load(bank_path, map_location="cpu")
    _apply_prediction_history_bank(dataset, bank)
    return bank


def merge_unrelaxed_validation_predictions(
    output_dir, validation_name, num_tasks, expected_frames, split="val"
):
    """Merge immutable per-rank raw predictions and derive all metrics offline."""
    prediction_dir = os.path.join(output_dir, f"{split}_predictions", validation_name)
    rows = []
    seen = set()
    for rank in range(num_tasks):
        part_path = os.path.join(prediction_dir, f"unrelaxed.rank{rank}.jsonl")
        with open(part_path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (str(row["video_id"]), int(row["frame_id"]))
                if key in seen:
                    raise RuntimeError(f"Duplicate {split} prediction: {key}")
                seen.add(key)
                rows.append(row)
    if len(rows) != expected_frames:
        raise RuntimeError(
            f"Expected {expected_frames} {split} predictions, found {len(rows)}"
        )
    rows.sort(key=lambda row: (str(row["video_id"]), int(row["frame_id"])))
    output_path = os.path.join(prediction_dir, "unrelaxed.jsonl")
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temporary_path, output_path)

    # Strict and relaxed scores are deterministic post-processing of the same
    # immutable unrelaxed predictions; model inference is never repeated.
    from evaluation_lungreco import evaluate, load_jsonl

    videos = load_jsonl(Path(output_path))

    metric_settings = (
        ("metrics_strict.json", False, 5),
        ("metrics_relaxed_k5.json", True, 5),
    )
    for filename, relaxed, relax_k in metric_settings:
        metrics = evaluate(videos, relaxed=relaxed, relax_k=relax_k)
        with open(os.path.join(prediction_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    print(f"Saved {len(rows)} immutable unrelaxed predictions to {output_path}")
    return output_path


def validation_prediction_part(output_dir, validation_name, rank, split="val"):
    return os.path.join(
        output_dir,
        f"{split}_predictions",
        validation_name,
        f"unrelaxed.rank{rank}.jsonl",
    )

def get_args():
    parser = argparse.ArgumentParser(
        "SurgVideoMAE fine-tuning and evaluation script for video phase recognition",
        add_help=False,
    )
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--stop_after_epoch",
        default=-1,
        type=int,
        help="Stop cleanly after this completed epoch while retaining the full schedule",
    )
    parser.add_argument(
        "--early_stopping_patience",
        default=0,
        type=int,
        help=(
            "Stop after this many consecutive completed epochs without a new "
            "best strict test video Accuracy; 0 disables early stopping"
        ),
    )
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument(
        "--save_every_epoch_ckpt",
        action="store_true",
        help="Save a resumable DeepSpeed checkpoint after every completed epoch",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/path/to/TimeSformer_divST_8x32_224_K400.pyth",
        type=str,
        metavar="Parameters",
        help="Name of parameters to load",
    )
    parser.add_argument("--input_size", default=224, type=int, help="videos input size")
    parser.add_argument(
        "--llava_model_path",
        default="liuhaotian/llava-v1.5-7b",
        help="Frozen LLaVA/Vicuna-7B checkpoint; use 'none' only for smoke tests",
    )
    parser.add_argument("--mcr_mask_ratio", default=0.4, type=float)
    parser.add_argument("--gaussian_sigma", default=3.0, type=float)
    parser.add_argument("--gaussian_history_length", default=64, type=int)
    parser.add_argument(
        "--feature_memory_gaussian_smoothing",
        action="store_true",
        help="Apply Gaussian smoothing to xLSTM feature memory before RFM",
    )
    parser.add_argument(
        "--disable_output_probability_smoothing",
        action="store_true",
        help="Use raw probabilities for histories and final online predictions",
    )
    parser.add_argument(
        "--phase_history_mask_probability",
        default=0.0,
        type=float,
        help="Independent masking probability for each phase-history token (paper RFM=0.4)",
    )
    parser.add_argument(
        "--transition_history_length",
        default=3,
        type=int,
        help="Number of most recent predicted phase transitions used by MCR",
    )
    parser.add_argument(
        "--frame_phase_history",
        action="store_true",
        help="Use every preceding phase frame (including OO) rather than distinct transitions",
    )
    parser.add_argument(
        "--llava_prompt_cache_size",
        default=0,
        type=int,
        help="LRU entries for exact frozen-LLaVA prompt embeddings (0 disables)",
    )
    parser.add_argument(
        "--prediction_history_bank",
        action="store_true",
        help="Train CoST from the previous epoch's smoothed predicted transitions",
    )
    parser.add_argument(
        "--prediction_history_distinct_non_other",
        action="store_true",
        help="Use the latest distinct predicted phases while excluding class 6",
    )
    parser.add_argument(
        "--freeze_prediction_history_bank",
        action="store_true",
        help="Keep the bootstrapped checkpoint history bank fixed across epochs",
    )
    parser.add_argument(
        "--static_train_history_bank",
        default="",
        type=str,
        help="Frozen stage-1 teacher history bank reused without regeneration",
    )
    parser.add_argument(
        "--gt_train_history_bank",
        action="store_true",
        help="Build frozen strictly-causal train histories directly from clean GT labels",
    )
    parser.add_argument(
        "--phase_history_replace_ratio",
        default=0.0,
        type=float,
        help="Conditional probability of replacing the selected history token instead of masking it",
    )
    parser.add_argument(
        "--export_history_bank_only",
        action="store_true",
        help="Build a history bank from --finetune and exit before training",
    )
    parser.add_argument(
        "--static_test_history_predictions",
        default="",
        type=str,
        help="Saved smoothed JSONL predictions used as frozen test histories",
    )
    parser.add_argument(
        "--static_val_history_predictions",
        default="",
        type=str,
        help="Saved smoothed JSONL predictions used as frozen validation histories",
    )
    parser.add_argument(
        "--select_on_val_only",
        action="store_true",
        help=(
            "Select checkpoint-best and early-stop on strict validation video "
            "Accuracy, without evaluating the test split during training"
        ),
    )
    parser.add_argument(
        "--static_finetune_history_predictions",
        default="",
        type=str,
        help="Saved JSONL predictions used as frozen histories for a diagnostic fine-tuning split",
    )
    parser.add_argument(
        "--include_text_final_fusion",
        action="store_true",
        help="Restore the archived local+memory+text final fusion architecture",
    )
    parser.add_argument(
        "--finetune_test_video_id",
        default="",
        type=str,
        help="Diagnostic only: train on one test video with frozen cached histories",
    )
    parser.add_argument(
        "--finetune_val_split",
        action="store_true",
        help="Diagnostic only: fine-tune on the validation videos with frozen cached histories",
    )
    parser.add_argument(
        "--finetune_sample_interval",
        default=1,
        type=int,
        help="Keep one diagnostic fine-tuning target every N frames in each video",
    )
    parser.add_argument(
        "--train_video_ids",
        default="",
        type=str,
        help="Comma-separated LungRes80 training video IDs; empty uses all train videos",
    )
    parser.add_argument(
        "--train_sample_interval",
        default=1,
        type=int,
        help="Keep one training target every N frames per video before shuffled sampling",
    )
    parser.add_argument(
        "--prediction_history_dropout",
        default=0.2,
        type=float,
        help="Probability of replacing a prediction-bank history with an empty history",
    )
    parser.add_argument(
        "--prediction_history_corruption",
        default=0.0,
        type=float,
        help="Probability of deleting or replacing one non-empty history entry",
    )
    parser.add_argument(
        "--history_text_scale",
        default=0.1,
        type=float,
        help="Fixed scale applied to the normalized history token inside CoST",
    )
    parser.add_argument(
        "--freeze_except_history_fusion",
        action="store_true",
        help="Train only CoST history attention/norms, final fusion MLP, and classifier",
    )
    parser.add_argument(
        "--freeze_history_fusion",
        action="store_true",
        help="Freeze the same CoST history attention/norms, final fusion MLP, and classifier while training the remaining model",
    )
    parser.add_argument("--initial_best_test_accuracy", default=float("-inf"), type=float)
    parser.add_argument("--initial_best_test_epoch", default=-1, type=int)
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Activation checkpoint visual encoder blocks without changing model math",
    )
    parser.add_argument(
        "--disable_lr_scale",
        action="store_true",
        help="Use --lr literally instead of scaling it from a batch size of 64",
    )

    parser.add_argument(
        "--fc_drop_rate",
        type=float,
        default=0.5,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--attn_drop_rate",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Attention dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--disable_eval_during_finetuning", action="store_true", default=False
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=(0.9, 0.999),
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4/1e-3)",
    )
    parser.add_argument("--layer_decay", type=float, default=0.75)

    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports, default 5",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m7-n4-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--short_side_size", type=int, default=224)

    # Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0, default 0.8.",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0, default 1.0.",
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/path/to/LungRes80",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--eval_data_path",
        default="/path/to/LungRes80",
        type=str,
        help="dataset path for evaluation",
    )
    parser.add_argument(
        "--nb_classes", default=7, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--imagenet_default_mean_and_std", default=True, action="store_true"
    )
    parser.add_argument(
        "--paper_augmentation",
        action="store_true",
        help="Use only resize and random horizontal flip as reported for LungReco",
    )

    parser.add_argument(
        "--data_strategy", type=str, default="online"
    )  # online/offline
    parser.add_argument(
        "--output_mode", type=str, default="key_frame"
    )  # key_frame/all_frame
    parser.add_argument("--cut_black", action="store_true")  # True/False
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument(
        "--sampling_rate", type=int, default=4
    )  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
    parser.add_argument(
        "--data_set",
        default="LungRes80",
        choices=["LungRes80"],
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--data_fps",
        default="1fps",
        choices=["", "5fps", "1fps"],
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/LungRes80",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="./results/LungRes80/log",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--eval", action="store_true", default=False, help="Perform evaluation only"
    )
    parser.add_argument(
        "--eval_val",
        action="store_true",
        default=False,
        help="Evaluate the validation split only, using the resumed checkpoint",
    )
    parser.add_argument(
        "--eval_lungreco_test",
        action="store_true",
        default=False,
        help="Run the LungReco test evaluator once and persist strict/k5 outputs",
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--debug_train_steps",
        default=0,
        type=int,
        help="Limit optimizer steps per epoch for smoke tests only (0 disables)",
    )
    parser.add_argument(
        "--debug_skip_epoch_test",
        action="store_true",
        help="Skip per-epoch test for smoke tests only",
    )
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    
    parser.add_argument("--enable_deepspeed", action="store_true", default=False)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use native BF16 in DeepSpeed (recommended for LungReco/xLSTM)",
    )

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    if not 0.0 <= args.prediction_history_dropout <= 1.0:
        raise ValueError("prediction_history_dropout must be in [0, 1]")
    if not 0.0 <= args.prediction_history_corruption <= 1.0:
        raise ValueError("prediction_history_corruption must be in [0, 1]")
    if not 0.0 <= args.history_text_scale <= 1.0:
        raise ValueError("history_text_scale must be in [0, 1]")
    if not 0.0 <= args.phase_history_replace_ratio <= 1.0:
        raise ValueError("phase_history_replace_ratio must be in [0, 1]")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be non-negative")
    if args.gt_train_history_bank and not args.prediction_history_bank:
        raise ValueError("--gt_train_history_bank requires --prediction_history_bank")
    if args.gt_train_history_bank and args.static_train_history_bank:
        raise ValueError("GT and static teacher train-history banks are mutually exclusive")
    print("===========    ", "Init Distribution", "    ===========")
    utils.init_distributed_mode(args)

    if ds_init is not None:
        print("===========    ", "Use Deepspeed", "    ===========")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        utils.create_ds_config(args)

    print(args)
    if args.sampling_rate == 0:
        frame_manner = "Exponential_Stride"
    elif args.sampling_rate == -1:
        frame_manner = "Random_Stride"
    elif args.sampling_rate == -2:
        frame_manner = "Incremental_Stride"
    else:
        frame_manner = "Fixed_Stride_" + str(args.sampling_rate)

    args.output_dir = os.path.join(
        args.output_dir,
        "_".join(
            [
                args.model,
                args.data_set,
                str(args.lr),
                str(args.layer_decay),
                args.data_strategy,
                args.output_mode,
                "frame" + str(args.num_frames),
                frame_manner,
            ]
        ),
    )

    args.log_dir = os.path.join(args.output_dir, "log")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not (args.eval or args.eval_val or args.eval_lungreco_test):
        txt_file = open(os.path.join(args.output_dir, "hyerparamter.txt"), "w")
        txt_file.write(str(args))
    else:
        txt_file = open(os.path.join(args.output_dir, "val_hyerparamter.txt"), "w")
        txt_file.write(str(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(
        is_train=True, test_mode=False, fps=args.data_fps, args=args
    )

    # 是否在训练时在验证集上测试性能
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(
            is_train=False, test_mode=False, fps=args.data_fps, args=args
        )
    dataset_test, _ = build_dataset(
        is_train=False, test_mode=True, fps=args.data_fps, args=args
    )

    if args.train_video_ids:
        requested_train_videos = [
            value.strip().zfill(2)
            for value in args.train_video_ids.split(",")
            if value.strip()
        ]
        if len(requested_train_videos) != len(set(requested_train_videos)):
            raise ValueError("--train_video_ids contains duplicate video IDs")
        available_train_videos = {
            str(sample["video_id"]).zfill(2)
            for sample in dataset_train.dataset_samples
        }
        missing_train_videos = sorted(
            set(requested_train_videos) - available_train_videos
        )
        if missing_train_videos:
            raise ValueError(
                "Requested training videos are not in the LungRes80 train split: "
                + ",".join(missing_train_videos)
            )
        requested_train_video_set = set(requested_train_videos)
        dataset_train.dataset_samples = [
            sample
            for sample in dataset_train.dataset_samples
            if str(sample["video_id"]).zfill(2) in requested_train_video_set
        ]
        retained_train_videos = sorted(
            {
                str(sample["video_id"]).zfill(2)
                for sample in dataset_train.dataset_samples
            }
        )
        if len(retained_train_videos) != len(requested_train_videos):
            raise RuntimeError("Filtered training split lost one or more requested videos")
        print(
            "Restricted LungRes80 training split to "
            f"{len(retained_train_videos)} videos and "
            f"{len(dataset_train.dataset_samples)} frames: "
            + ",".join(retained_train_videos)
        )
    if args.train_sample_interval <= 0:
        raise ValueError("--train_sample_interval must be positive")
    if args.train_sample_interval > 1:
        positions_by_video = {}
        selected_indices = []
        for sample_index, sample in enumerate(dataset_train.dataset_samples):
            video_id = str(sample["video_id"]).zfill(2)
            position = positions_by_video.get(video_id, 0)
            if position % int(args.train_sample_interval) == 0:
                selected_indices.append(sample_index)
            positions_by_video[video_id] = position + 1
        # Keep the full frame table intact: _video_batch_loader indexes it to
        # fetch the target frame's preceding clip.  Only target indices should
        # be sparse and shuffled.
        dataset_train = torch.utils.data.Subset(dataset_train, selected_indices)
        print(
            "Sparse LungRes80 training targets: interval="
            f"{args.train_sample_interval}, "
            f"retained_frames={len(dataset_train)}, "
            f"videos={len(positions_by_video)}; sampler remains shuffled"
        )

    if args.static_test_history_predictions:
        _apply_prediction_json_histories(
            dataset_test,
            args.static_test_history_predictions,
            max_phases=args.transition_history_length,
            excluded_phase_ids=(6,),
            prediction_field=(
                "raw_prediction"
                if args.disable_output_probability_smoothing
                else "prediction"
            ),
        )
    if args.static_val_history_predictions:
        _apply_prediction_json_histories(
            dataset_val,
            args.static_val_history_predictions,
            max_phases=args.transition_history_length,
            excluded_phase_ids=(6,),
            prediction_field=(
                "raw_prediction"
                if args.disable_output_probability_smoothing
                else "prediction"
            ),
        )
    if args.finetune_val_split and args.finetune_test_video_id:
        raise ValueError(
            "Choose either validation-split fine-tuning or single-test-video fine-tuning"
        )
    if args.finetune_sample_interval <= 0:
        raise ValueError("--finetune_sample_interval must be positive")
    if args.finetune_val_split:
        if not args.static_finetune_history_predictions:
            raise ValueError(
                "Validation-split fine-tuning requires frozen validation predictions"
            )
        _apply_prediction_json_histories(
            dataset_val,
            args.static_finetune_history_predictions,
            max_phases=args.transition_history_length,
            excluded_phase_ids=(6,),
            prediction_field=(
                "raw_prediction"
                if args.disable_output_probability_smoothing
                else "prediction"
            ),
        )
        val_train_dataset = copy.copy(dataset_val)
        val_train_dataset.mode = "train"
        val_train_dataset.aug = True
        val_train_dataset.rand_erase = bool(args.reprob > 0)
        interval = int(args.finetune_sample_interval)
        positions_by_video = {}
        selected_indices = []
        for sample_index, sample in enumerate(dataset_val.dataset_samples):
            video_id = str(sample["video_id"])
            position = positions_by_video.get(video_id, 0)
            if position % interval == 0:
                selected_indices.append(sample_index)
            positions_by_video[video_id] = position + 1
        dataset_train = torch.utils.data.Subset(
            val_train_dataset, selected_indices
        )
        print(
            f"Diagnostic fine-tuning split: all validation videos, "
            f"interval={interval}, {len(dataset_train)} frames, shuffled sampler"
        )
    if args.finetune_test_video_id:
        if not args.static_test_history_predictions:
            raise ValueError(
                "Single-test-video fine-tuning requires static test predictions"
            )
        selected_video = str(args.finetune_test_video_id).zfill(2)
        test_video_train_dataset = copy.copy(dataset_test)
        test_video_train_dataset.mode = "train"
        test_video_train_dataset.aug = True
        test_video_train_dataset.dataset_samples = [
            sample
            for sample in dataset_test.dataset_samples
            if str(sample["video_id"]).zfill(2) == selected_video
        ]
        if not test_video_train_dataset.dataset_samples:
            raise ValueError(f"Test video {selected_video} was not found")
        dataset_train = test_video_train_dataset
        print(
            f"Diagnostic fine-tuning split: test video {selected_video}, "
            f"{len(dataset_train)} frames, shuffled sampler"
        )

    print("Train Dataset Length: ", len(dataset_train))
    print("Val Dataset Length: ", len(dataset_val))
    print("Test Dataset Length: ", len(dataset_test))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Distribute Sampler For Val/Test")
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 顺序采样
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        print("Sequential Sampler For Val/Test")

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        print("Log dir:", args.log_dir)
    else:
        log_writer = None

    collate_func = None

    # Match the archived baseline's shuffled DistributedSampler exactly.
    # LungReco receives GT transition context as per-sample metadata during
    # training, so it does not require chronological batches.
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if dataset_val is not None:
        val_batch_size = (
            args.batch_size if args.model == "lungreco" else int(10 * args.batch_size)
        )
        if args.model == "lungreco":
            val_batch_sampler = VideoSequentialDistributedBatchSampler(
                dataset_val, val_batch_size, num_tasks, global_rank
            )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_sampler=val_batch_sampler,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )
        else:
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                sampler=sampler_val,
                batch_size=val_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
    else:
        data_loader_val = None

    if dataset_test is not None:
        test_batch_size = (
            args.batch_size if args.model == "lungreco" else int(8 * args.batch_size)
        )
        if args.model == "lungreco":
            test_batch_sampler = VideoSequentialDistributedBatchSampler(
                dataset_test, test_batch_size, num_tasks, global_rank
            )
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_sampler=test_batch_sampler,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                sampler=sampler_test,
                batch_size=test_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
    else:
        data_loader_test = None

    # 训练Trick，有效显著的数据增强效果，参见：https://blog.csdn.net/sophicchen/article/details/120432083
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model_kwargs = dict(
        pretrained=True,
        pretrain_path=args.pretrained_path,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
    )
    if args.model == "lungreco":
        model_kwargs.update(
            llava_model_path=args.llava_model_path,
            mcr_mask_ratio=args.mcr_mask_ratio,
            gaussian_sigma=args.gaussian_sigma,
            gaussian_history_length=args.gaussian_history_length,
            transition_history_length=args.transition_history_length,
            history_text_scale=args.history_text_scale,
            feature_memory_gaussian_smoothing=(
                args.feature_memory_gaussian_smoothing
            ),
            smooth_output_probabilities=(
                not args.disable_output_probability_smoothing
            ),
            phase_history_mask_probability=(
                args.phase_history_mask_probability
            ),
            phase_history_replace_ratio=args.phase_history_replace_ratio,
            llava_prompt_cache_size=args.llava_prompt_cache_size,
            use_checkpoint=args.use_checkpoint,
            use_text_cost_fusion=False,
            include_text_final_fusion=args.include_text_final_fusion,
        )
    model = create_model(args.model, **model_kwargs)
    txt_file.write(str(model))
    txt_file.close()

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (
        args.num_frames,
        args.input_size // patch_size[0],
        args.input_size // patch_size[1],
    )
    print("Window size: = %s" % str(args.window_size))
    args.patch_size = patch_size

    # 加载预训练参数，并且根据策略调整Patch_embedding，可直接加载基于VideoMAE预训练参数
    # 也可以加载官方的VIT的预训练参数
    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith("backbone."):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith("encoder."):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if "pos_embed" in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(
                (
                    (pos_embed_checkpoint.shape[-2] - num_extra_tokens)
                    // (args.num_frames)
                )
                ** 0.5
            )
            # height (== width) for the new position embedding
            new_size = int(
                (num_patches // (args.num_frames))
                ** 0.5
            )
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1,
                    args.num_frames,
                    orig_size,
                    orig_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.reshape(
                    -1, orig_size, orig_size, embedding_size
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1,
                    args.num_frames,
                    new_size,
                    new_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model["pos_embed"] = new_pos_embed

        if 'time_embed' in checkpoint_model and args.num_frames != checkpoint_model['time_embed'].size(1):
            time_embed = checkpoint_model['time_embed'].transpose(1, 2).float()
            new_time_embed = F.interpolate(time_embed, size=(args.num_frames), mode='nearest')
            checkpoint_model['time_embed'] = new_time_embed.transpose(1, 2)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    history_fusion_prefixes = (
        "cost.spatial_norm.",
        "cost.spatial_attn.",
        "cost.output_norm.",
        "fusion.",
        "head.",
    )
    if args.freeze_except_history_fusion and args.freeze_history_fusion:
        raise ValueError(
            "--freeze_except_history_fusion and --freeze_history_fusion are mutually exclusive"
        )
    if args.freeze_except_history_fusion:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(history_fusion_prefixes)
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        trainable_count = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            "History-fusion-only fine-tuning: "
            f"{trainable_count:,} trainable parameters"
        )
        print("Trainable parameter tensors: " + ", ".join(trainable_names))
    elif args.freeze_history_fusion:
        frozen_names = []
        for name, parameter in model.named_parameters():
            if name.startswith(history_fusion_prefixes):
                parameter.requires_grad = False
                frozen_names.append(name)
        frozen_count = sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if name.startswith(history_fusion_prefixes)
        )
        print(
            "Frozen trained history-fusion path: "
            f"{frozen_count:,} parameters"
        )
        print("Frozen parameter tensors: " + ", ".join(frozen_names))

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    if args.debug_train_steps > 0:
        num_training_steps_per_epoch = min(
            num_training_steps_per_epoch, args.debug_train_steps
        )
    if not args.disable_lr_scale:
        args.lr = args.lr * total_batch_size / 64
        args.min_lr = args.min_lr * total_batch_size / 64
        args.warmup_lr = args.warmup_lr * total_batch_size / 64
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()

    if args.layer_decay == 0.1:
        assigner = LayerDecayValueAssigner(
            [args.layer_decay] * (num_layers + 1) + [1.0]
        )
    elif args.layer_decay < 1.0:  # 沿层以几何方式降低学习率
        assigner = LayerDecayValueAssigner(
            list(
                args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
            )
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model,
            args.weight_decay,
            skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print(
            "model.gradient_accumulation_steps() = %d"
            % model.gradient_accumulation_steps()
        )
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=None,
    )

    prediction_history_bank = None
    if args.model == "lungreco" and args.prediction_history_bank:
        diagnostic_static_history = bool(
            args.finetune_test_video_id or args.finetune_val_split
        )
        if args.gt_train_history_bank:
            prediction_history_bank = _build_gt_history_bank(
                dataset_train,
                max_phases=args.transition_history_length,
                excluded_phase_ids=(6,),
                fps=1.0,
                frame_history=args.frame_phase_history,
            )
            _apply_prediction_history_bank(dataset_train, prediction_history_bank)
            bank_dir = Path(args.output_dir) / "history_banks"
            if utils.is_main_process():
                bank_dir.mkdir(parents=True, exist_ok=True)
                torch.save(prediction_history_bank, bank_dir / "history_bank_gt_clean.pt")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            print(
                "Built immutable strictly-causal GT history bank for "
                f"{len(prediction_history_bank['frame_ids'])} training frames"
            )
        elif args.static_train_history_bank:
            prediction_history_bank = torch.load(
                args.static_train_history_bank, map_location="cpu"
            )
            _apply_prediction_history_bank(dataset_train, prediction_history_bank)
            print(
                "Loaded immutable stage-1 teacher history bank from "
                f"{args.static_train_history_bank}; "
                f"source_epoch={prediction_history_bank['source_epoch']}"
            )
        elif diagnostic_static_history:
            prediction_history_bank = {
                "source_epoch": 1,
                "frozen_split": (
                    f"test_video_{str(args.finetune_test_video_id).zfill(2)}"
                    if args.finetune_test_video_id
                    else "validation"
                ),
            }
            print(
                "Using frozen epoch-1 histories already attached to diagnostic "
                f"split {prediction_history_bank['frozen_split']}"
            )
        else:
            bank_path = _latest_prediction_history_bank(args.output_dir)
        if (
            not args.gt_train_history_bank
            and not args.static_train_history_bank
            and not diagnostic_static_history
            and bank_path is not None
        ):
            prediction_history_bank = torch.load(bank_path, map_location="cpu")
            _apply_prediction_history_bank(dataset_train, prediction_history_bank)
            print(
                f"Loaded prediction history bank from {bank_path}; "
                f"source_epoch={prediction_history_bank['source_epoch']}"
            )
        elif (
            not args.gt_train_history_bank
            and not args.static_train_history_bank
            and not diagnostic_static_history
            and args.start_epoch > 0
        ):
            # A resumed checkpoint already represents the previous epoch. Run
            # one update-free train-set pass so the very first resumed epoch
            # can consume predicted histories instead of wasting an epoch to
            # create them.
            # Build the epoch-1 bank with deterministic inference transforms,
            # while preserving the exact train split and per-sample indices.
            bootstrap_dataset = copy.copy(dataset_train)
            bootstrap_dataset.mode = "val"
            bootstrap_dataset.data_transform = dataset_val.data_transform
            bootstrap_sampler = torch.utils.data.DistributedSampler(
                bootstrap_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
                drop_last=False,
            )
            bootstrap_loader = torch.utils.data.DataLoader(
                bootstrap_dataset,
                sampler=bootstrap_sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                collate_fn=collate_func,
            )
            wrapped_model = getattr(model, "module", model)
            # The teacher has no phase history during this export pass, so
            # reproduce its history-free inference path exactly. The final
            # fixed text fusion remains controlled by the model architecture.
            wrapped_model.use_text_cost_fusion = False
            print(
                "No prediction history bank found; bootstrapping it from "
                f"the resumed checkpoint for source_epoch={args.start_epoch - 1}"
            )
            bootstrap_records = collect_prediction_history_records(
                model, bootstrap_loader, device
            )
            prediction_history_bank = _build_prediction_history_bank(
                dataset_train,
                bootstrap_records,
                args.output_dir,
                args.start_epoch - 1,
                num_tasks,
                args.gaussian_sigma,
                args.gaussian_history_length,
                args.transition_history_length,
                distinct_non_other=args.prediction_history_distinct_non_other,
            )

    if args.export_history_bank_only:
        if prediction_history_bank is None:
            raise ValueError(
                "--export_history_bank_only requires --prediction_history_bank "
                "and a resumed --finetune checkpoint"
            )
        print("History bank export completed; exiting before training")
        return

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + ".txt")
        test_stats = final_phase_test(data_loader_test, model, device, preds_file)
        print("Save Files: ", preds_file)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1, final_top5 = merge(args.output_dir, num_tasks)
            print(
                f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
            )
            log_stats = {"Final top-1": final_top1, "Final Top-5": final_top5}
            if args.output_dir and utils.is_main_process():
                with open(
                    os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)

    if args.eval_val:
        prediction_part = None
        if args.model == "lungreco":
            prediction_part = validation_prediction_part(
                args.output_dir, "checkpoint_eval", global_rank
            )
        val_stats = validation_one_epoch(
            data_loader_val, model, device, prediction_file=prediction_part
        )
        torch.distributed.barrier()
        if global_rank == 0:
            if prediction_part is not None:
                merge_unrelaxed_validation_predictions(
                    args.output_dir,
                    "checkpoint_eval",
                    num_tasks,
                    len(dataset_val),
                )
            print(
                f"Validation checkpoint metrics on {len(dataset_val)} frames: "
                f"Acc@1 {val_stats['acc1']:.4f}, Acc@5 {val_stats['acc5']:.4f}, "
                f"loss {val_stats['loss']:.6f}"
            )
        exit(0)

    if args.eval_lungreco_test:
        if args.model != "lungreco":
            raise ValueError("--eval_lungreco_test is only valid for --model lungreco")
        wrapped_model = getattr(model, "module", model)
        # History fusion can consume either a frozen external history or the
        # model's own causal online predictions. The latter is selected by
        # --prediction_history_bank without --static_test_history_predictions.
        wrapped_model.use_text_cost_fusion = bool(args.prediction_history_bank)
        prediction_part = validation_prediction_part(
            args.output_dir, "final", global_rank, split="test"
        )
        test_stats = validation_one_epoch(
            data_loader_test,
            model,
            device,
            prediction_file=prediction_part,
            split_name="Test",
            static_prediction_history=bool(args.static_test_history_predictions),
        )
        torch.distributed.barrier()
        if global_rank == 0:
            test_output_path = merge_unrelaxed_validation_predictions(
                args.output_dir,
                "final",
                num_tasks,
                len(dataset_test),
                split="test",
            )
            print(
                "Final LungReco test outputs saved to "
                f"{os.path.dirname(test_output_path)}; "
                f"frame Acc@1 {test_stats['acc1']:.4f}"
            )
        torch.distributed.barrier()
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_epoch = 0
    max_test_accuracy = float(args.initial_best_test_accuracy)
    max_test_epoch = int(args.initial_best_test_epoch)
    epochs_without_test_improvement = 0
    for epoch in range(args.start_epoch, args.epochs):
        history_active = bool(
            args.model == "lungreco"
            and args.prediction_history_bank
            and prediction_history_bank is not None
        )
        if args.model == "lungreco":
            wrapped_model = getattr(model, "module", model)
            wrapped_model.use_text_cost_fusion = history_active
            print(
                f"Epoch {epoch} prediction history: "
                f"active={history_active}, dropout="
                f"{args.prediction_history_dropout if history_active else 0.0:.2f}, "
                f"corruption={args.prediction_history_corruption if history_active else 0.0:.2f}, "
                f"text_scale={args.history_text_scale:.2f}"
            )
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        history_prediction_records = {"indices": [], "probabilities": []}
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            None,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            prediction_history_active=history_active,
            history_dropout=(
                args.prediction_history_dropout if history_active else 0.0
            ),
            history_corruption_prob=(
                args.prediction_history_corruption if history_active else 0.0
            ),
            history_prediction_records=(
                history_prediction_records
                if (
                    args.model == "lungreco"
                    and args.prediction_history_bank
                    and not args.freeze_prediction_history_bank
                )
                else None
            ),
        )
        if (
            args.model == "lungreco"
            and args.prediction_history_bank
            and not args.freeze_prediction_history_bank
        ):
            prediction_history_bank = _build_prediction_history_bank(
                dataset_train,
                history_prediction_records,
                args.output_dir,
                epoch,
                num_tasks,
                args.gaussian_sigma,
                args.gaussian_history_length,
                args.transition_history_length,
                distinct_non_other=args.prediction_history_distinct_non_other,
            )
        if args.model == "lungreco" and utils.is_main_process():
            wrapped_model = getattr(model, "module", model)
            text_encoder = wrapped_model.mcr.text_encoder
            hits = int(text_encoder.prompt_cache_hits)
            misses = int(text_encoder.prompt_cache_misses)
            total = hits + misses
            if total:
                print(
                    f"Frozen LLaVA prompt cache: hits={hits}, misses={misses}, "
                    f"hit_rate={100.0 * hits / total:.2f}%"
                )
        if args.model != "lungreco" and args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=None,
                )
        if data_loader_val is not None and args.model != "lungreco":
            validation_name = f"epoch_{epoch:03d}"
            prediction_part = None
            if args.model == "lungreco":
                prediction_part = validation_prediction_part(
                    args.output_dir, validation_name, global_rank
                )
            test_stats = validation_one_epoch(
                data_loader_val, model, device, prediction_file=prediction_part
            )
            torch.distributed.barrier()
            if global_rank == 0 and prediction_part is not None:
                merge_unrelaxed_validation_predictions(
                    args.output_dir,
                    validation_name,
                    num_tasks,
                    len(dataset_val),
                )
            torch.distributed.barrier()
            print(
                f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%"
            )
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                max_epoch = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=None,
                    )

            print(
                f"Max accuracy: {max_accuracy:.2f}%" + "   Max Epoch: " + str(max_epoch)
            )
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats["acc1"], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats["acc5"], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats["loss"], head="perf", step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if (
            args.model == "lungreco"
            and args.select_on_val_only
            and data_loader_val is not None
        ):
            if utils.is_main_process():
                print(
                    "Stage validation history mode: "
                    + (
                        "frozen external histories"
                        if args.static_val_history_predictions
                        else "current model causal online self-history"
                    )
                )
            validation_name = f"epoch_{epoch:03d}"
            val_prediction_part = validation_prediction_part(
                args.output_dir, validation_name, global_rank, split="val"
            )
            val_epoch_stats = validation_one_epoch(
                data_loader_val,
                model,
                device,
                prediction_file=val_prediction_part,
                split_name="Val",
                static_prediction_history=bool(
                    args.static_val_history_predictions
                ),
            )
            torch.distributed.barrier()
            val_video_accuracy = val_epoch_stats["acc1"]
            if global_rank == 0:
                val_output_path = merge_unrelaxed_validation_predictions(
                    args.output_dir,
                    validation_name,
                    num_tasks,
                    len(dataset_val),
                    split="val",
                )
                with open(
                    os.path.join(
                        os.path.dirname(val_output_path), "metrics_strict.json"
                    ),
                    "r",
                    encoding="utf-8",
                ) as handle:
                    val_video_accuracy = float(json.load(handle)["accuracy"])
            accuracy_tensor = torch.tensor(
                val_video_accuracy if global_rank == 0 else 0.0,
                dtype=torch.float64,
                device=device,
            )
            if args.distributed:
                torch.distributed.broadcast(accuracy_tensor, src=0)
            val_video_accuracy = float(accuracy_tensor.item())
            torch.distributed.barrier()
            print(
                f"Accuracy of the network on the {len(dataset_val)} val frames: "
                f"frame Acc@1 {val_epoch_stats['acc1']:.1f}%, "
                f"video Accuracy {val_video_accuracy:.2f}%"
            )
            log_stats.update(
                {f"val_{key}": value for key, value in val_epoch_stats.items()}
            )
            log_stats["val_video_accuracy"] = val_video_accuracy
            val_accuracy_improved = val_video_accuracy > max_test_accuracy
            if val_accuracy_improved:
                max_test_accuracy = val_video_accuracy
                max_test_epoch = epoch
                epochs_without_test_improvement = 0
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=None,
                    )
            else:
                epochs_without_test_improvement += 1
            print(
                f"Best val video Accuracy: {max_test_accuracy:.2f}%   "
                f"Best Epoch: {max_test_epoch}"
            )
            if args.early_stopping_patience > 0:
                print(
                    "Early stopping: epochs_without_improvement="
                    f"{epochs_without_test_improvement}/"
                    f"{args.early_stopping_patience}"
                )
            log_stats["max_val_video_accuracy"] = max_test_accuracy
            log_stats["max_val_epoch"] = max_test_epoch
            log_stats["epochs_without_test_improvement"] = (
                epochs_without_test_improvement
            )

        if (
            data_loader_test is not None
            and not args.debug_skip_epoch_test
            and not args.select_on_val_only
        ):
            test_name = f"epoch_{epoch:03d}"
            test_prediction_part = None
            if args.model == "lungreco":
                test_prediction_part = validation_prediction_part(
                    args.output_dir,
                    test_name,
                    global_rank,
                    split="test",
                )
            epoch_test_stats = validation_one_epoch(
                data_loader_test,
                model,
                device,
                prediction_file=test_prediction_part,
                split_name="Test",
                static_prediction_history=bool(
                    args.static_test_history_predictions
                ),
            )
            torch.distributed.barrier()
            test_video_accuracy = epoch_test_stats["acc1"]
            if global_rank == 0 and test_prediction_part is not None:
                test_output_path = merge_unrelaxed_validation_predictions(
                    args.output_dir,
                    test_name,
                    num_tasks,
                    len(dataset_test),
                    split="test",
                )
                with open(
                    os.path.join(os.path.dirname(test_output_path), "metrics_strict.json"),
                    "r",
                    encoding="utf-8",
                ) as handle:
                    test_video_accuracy = float(json.load(handle)["accuracy"])
            accuracy_tensor = torch.tensor(
                test_video_accuracy if global_rank == 0 else 0.0,
                dtype=torch.float64,
                device=device,
            )
            if args.distributed:
                torch.distributed.broadcast(accuracy_tensor, src=0)
            test_video_accuracy = float(accuracy_tensor.item())
            torch.distributed.barrier()
            print(
                f"Accuracy of the network on the {len(dataset_test)} test frames: "
                f"frame Acc@1 {epoch_test_stats['acc1']:.1f}%, "
                f"video Accuracy {test_video_accuracy:.2f}%"
            )
            log_stats.update(
                {f"test_{key}": value for key, value in epoch_test_stats.items()}
            )
            log_stats["test_video_accuracy"] = test_video_accuracy
            test_accuracy_improved = bool(
                args.model == "lungreco"
                and test_video_accuracy > max_test_accuracy
            )
            if test_accuracy_improved:
                max_test_accuracy = test_video_accuracy
                max_test_epoch = epoch
                epochs_without_test_improvement = 0
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=None,
                    )
            elif args.model == "lungreco":
                epochs_without_test_improvement += 1
            if args.model == "lungreco":
                print(
                    f"Best test video Accuracy: {max_test_accuracy:.2f}%   "
                    f"Best Epoch: {max_test_epoch}"
                )
                if args.early_stopping_patience > 0:
                    print(
                        "Early stopping: epochs_without_improvement="
                        f"{epochs_without_test_improvement}/"
                        f"{args.early_stopping_patience}"
                    )
                log_stats["max_test_video_accuracy"] = max_test_accuracy
                log_stats["max_test_epoch"] = max_test_epoch
                log_stats["epochs_without_test_improvement"] = (
                    epochs_without_test_improvement
                )
            if log_writer is not None:
                log_writer.update(
                    test_acc1=epoch_test_stats["acc1"], head="perf", step=epoch
                )
                log_writer.update(
                    test_acc5=epoch_test_stats["acc5"], head="perf", step=epoch
                )
                log_writer.update(
                    test_loss=epoch_test_stats["loss"], head="perf", step=epoch
                )
        if (
            args.model == "lungreco"
            and args.save_every_epoch_ckpt
            and args.output_dir
            and args.save_ckpt
        ):
            # Numbered checkpoints are written only after training, bank
            # rebuilding, and evaluation all complete successfully.
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_ema=None,
            )
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
        if args.stop_after_epoch >= 0 and epoch >= args.stop_after_epoch:
            print(f"Stopping cleanly after requested epoch {epoch}")
            break
        if (
            args.model == "lungreco"
            and args.early_stopping_patience > 0
            and epochs_without_test_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping after epoch "
                f"{epoch}: strict "
                f"{'validation' if args.select_on_val_only else 'test'} "
                "video Accuracy did not improve for "
                f"{epochs_without_test_improvement} consecutive epochs; "
                f"retaining checkpoint-best from epoch {max_test_epoch}"
            )
            break

    # LungReco validation or test predictions and strict/k=5 metrics have
    # already been persisted above. Avoid the legacy generic test path.
    if args.model == "lungreco":
        return

    # ==================================================================
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
    )
    preds_file = os.path.join(args.output_dir, str(global_rank) + ".txt")
    best_pretrained_path = os.path.join(args.output_dir, "checkpoint-best/mp_rank_00_model_states.pt")
    checkpoint = torch.load(best_pretrained_path, map_location="cpu")

    print("Load ckpt from %s" % best_pretrained_path)
    checkpoint_model = None
    for model_key in args.model_key.split("|"):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != state_dict[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith("backbone."):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith("encoder."):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )

    test_stats = final_phase_test(data_loader_test, model, device, preds_file)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1, final_top5 = merge(args.output_dir, num_tasks)
        print(
            f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
        )
        log_stats = {"Final top-1": final_top1, "Final Top-5": final_top5}
        if args.output_dir and utils.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts, ds_init = get_args()

    main(opts, ds_init)
