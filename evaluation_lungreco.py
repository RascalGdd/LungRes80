#!/usr/bin/env python3
"""Evaluate LungReco predictions with every metric reported in the paper."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PHASE_NAMES = (
    "Fissure Dissection",
    "Vein Dissection",
    "Artery Dissection",
    "Bronchus Dissection",
    "Lymph Node Dissection",
    "Lung Segment Resection",
    "Other Operations",
)


def collapse(sequence: np.ndarray) -> list[int]:
    return [int(sequence[i]) for i in range(len(sequence)) if i == 0 or sequence[i] != sequence[i - 1]]


def levenshtein(first: list[int], second: list[int]) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def segments(sequence: np.ndarray) -> list[tuple[int, int, int]]:
    result = []
    start = 0
    for index in range(1, len(sequence) + 1):
        if index == len(sequence) or sequence[index] != sequence[start]:
            result.append((int(sequence[start]), start, index))
            start = index
    return result


def segmental_f1_counts(
    prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> tuple[int, int, int]:
    predicted = segments(prediction)
    truth = segments(target)
    used: set[int] = set()
    true_positive = 0
    for label, start, end in predicted:
        best_iou, best_index = 0.0, None
        for index, (gt_label, gt_start, gt_end) in enumerate(truth):
            if index in used or label != gt_label:
                continue
            intersection = max(0, min(end, gt_end) - max(start, gt_start))
            union = max(end, gt_end) - min(start, gt_start)
            iou = intersection / union
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= threshold:
            used.add(best_index)
            true_positive += 1
    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - true_positive
    return true_positive, false_positive, false_negative


def segmental_f1(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    true_positive, false_positive, false_negative = segmental_f1_counts(
        prediction, target, threshold
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return 100.0 * (2 * true_positive / denominator if denominator else 1.0)


def adc(prediction: np.ndarray, target: np.ndarray) -> float:
    """Algorithm 1: wrong predicted groups / number of true groups."""
    wrong = 0
    for predicted_label, start, end in segments(prediction):
        if not np.all(target[start:end] == predicted_label):
            wrong += 1
    return wrong / max(1, len(segments(target)))


def relax_prediction(prediction: np.ndarray, target: np.ndarray, radius: int = 5) -> np.ndarray:
    """Apply the LungRes80 baseline boundary protocol used by TMRNet/Trans-SVNet.

    Around a GT transition ``a -> b``, only predictions equal to ``a`` or ``b``
    are accepted within the last ``radius`` frames of ``a`` and the first
    ``radius`` frames of ``b``. Other phase predictions remain errors.
    """
    prediction = prediction.copy()
    allowed = [{int(target[index])} for index in range(len(target))]
    transitions = np.flatnonzero(target[1:] != target[:-1]) + 1
    for boundary in transitions:
        previous_phase = int(target[boundary - 1])
        next_phase = int(target[boundary])
        left = max(0, boundary - radius)
        right = min(len(target), boundary + radius)
        for index in range(left, right):
            allowed[index] = {previous_phase, next_phase}
    for index, accepted_phases in enumerate(allowed):
        if int(prediction[index]) in accepted_phases:
            prediction[index] = target[index]
    return prediction


def phase_metrics(prediction: np.ndarray, target: np.ndarray, classes: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.full(classes, np.nan)
    recall = np.full(classes, np.nan)
    jaccard = np.full(classes, np.nan)
    for label in range(classes):
        if not np.any(target == label):
            continue
        true_positive = np.sum((prediction == label) & (target == label))
        false_positive = np.sum((prediction == label) & (target != label))
        false_negative = np.sum((prediction != label) & (target == label))
        if true_positive + false_positive:
            precision[label] = 100 * true_positive / (true_positive + false_positive)
        recall[label] = 100 * true_positive / (true_positive + false_negative)
        jaccard[label] = 100 * true_positive / (true_positive + false_positive + false_negative)
    return precision, recall, jaccard


def frame_phase_metrics(
    prediction: np.ndarray, target: np.ndarray, classes: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match sklearn's split-level metrics with ``zero_division=0``."""
    precision = np.zeros(classes, dtype=float)
    recall = np.zeros(classes, dtype=float)
    jaccard = np.zeros(classes, dtype=float)
    for label in range(classes):
        true_positive = np.sum((prediction == label) & (target == label))
        false_positive = np.sum((prediction == label) & (target != label))
        false_negative = np.sum((prediction != label) & (target == label))
        if true_positive + false_positive:
            precision[label] = 100 * true_positive / (true_positive + false_positive)
        if true_positive + false_negative:
            recall[label] = 100 * true_positive / (true_positive + false_negative)
        if true_positive + false_positive + false_negative:
            jaccard[label] = 100 * true_positive / (
                true_positive + false_positive + false_negative
            )
    return precision, recall, jaccard


def evaluate(
    videos: dict[str, tuple[np.ndarray, np.ndarray]], relaxed: bool, relax_k: int = 5
) -> dict[str, float | dict[str, float | None]]:
    values = defaultdict(list)
    phase_values = defaultdict(list)
    frame_predictions = []
    frame_targets = []
    segmental_totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in (0.10, 0.25, 0.50)
    }
    for prediction, target in videos.values():
        scored_prediction = (
            relax_prediction(prediction, target, radius=relax_k) if relaxed else prediction
        )
        precision, recall, jaccard = phase_metrics(scored_prediction, target)
        values["accuracy"].append(100 * np.mean(scored_prediction == target))
        phase_values["precision"].append(precision)
        phase_values["recall"].append(recall)
        phase_values["jaccard"].append(jaccard)
        values["adc"].append(adc(scored_prediction, target))
        predicted_steps, true_steps = collapse(scored_prediction), collapse(target)
        edit = 1 - levenshtein(predicted_steps, true_steps) / max(len(predicted_steps), len(true_steps), 1)
        values["segmental_edit"].append(100 * edit)
        frame_predictions.append(scored_prediction)
        frame_targets.append(target)
        for threshold, totals in segmental_totals.items():
            true_positive, false_positive, false_negative = segmental_f1_counts(
                scored_prediction, target, threshold
            )
            totals["tp"] += true_positive
            totals["fp"] += false_positive
            totals["fn"] += false_negative
    result = {key: float(np.mean(items)) for key, items in values.items()}
    for threshold, totals in segmental_totals.items():
        precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
        recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        key = int(round(100 * threshold))
        result[f"segmental_precision@{key}"] = 100.0 * precision
        result[f"segmental_recall@{key}"] = 100.0 * recall
        result[f"segmental_f1@{key}"] = 100.0 * f1
    for key, rows in phase_values.items():
        # Compute video mean per phase, then
        # mean over phases, excluding a phase when it is absent in that video.
        matrix = np.stack(rows)
        per_phase = {}
        for label, phase_name in enumerate(PHASE_NAMES):
            finite = matrix[:, label][np.isfinite(matrix[:, label])]
            per_phase[phase_name] = float(np.mean(finite)) if finite.size else None
        available = [value for value in per_phase.values() if value is not None]
        result[key] = float(np.mean(available))
        result[f"{key}_per_phase"] = per_phase

    # The cited baselines also report metrics after concatenating every frame
    # in the split. Keep these separate from the MATLAB-style video/phase mean.
    all_prediction = np.concatenate(frame_predictions)
    all_target = np.concatenate(frame_targets)
    frame_precision, frame_recall, frame_jaccard = frame_phase_metrics(
        all_prediction, all_target
    )
    result["frame_accuracy"] = float(100 * np.mean(all_prediction == all_target))
    for name, per_phase_values in (
        ("frame_precision_macro", frame_precision),
        ("frame_recall_macro", frame_recall),
        ("frame_jaccard_macro", frame_jaccard),
    ):
        result[name] = float(np.mean(per_phase_values))
        metric_name = name.removeprefix("frame_").removesuffix("_macro")
        result[f"frame_{metric_name}_per_phase"] = {
            phase_name: float(per_phase_values[label])
            for label, phase_name in enumerate(PHASE_NAMES)
        }
    result["relax_k"] = int(relax_k) if relaxed else 0
    return result


def load_jsonl(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = defaultdict(list)
    for line in path.read_text().splitlines():
        item = json.loads(line)
        rows[str(item["video_id"])].append(
            (int(item["frame_id"]), int(item["prediction"]), int(item["target"]))
        )
    videos = {}
    for video_id, items in rows.items():
        items.sort()
        frame_ids = [item[0] for item in items]
        if frame_ids != list(range(len(frame_ids))):
            raise ValueError(f"Non-contiguous frame IDs for video {video_id}")
        videos[video_id] = (
            np.asarray([item[1] for item in items]),
            np.asarray([item[2] for item in items]),
        )
    return videos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="JSONL with video_id/frame_id/prediction/target")
    parser.add_argument(
        "--relaxed", action="store_true", help="Apply the LungRes80 baseline relaxed-boundary protocol"
    )
    parser.add_argument(
        "--relax-k", type=int, default=5, help="Frames accepted on each side of a GT transition (default: 5)"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(load_jsonl(args.predictions), args.relaxed, relax_k=args.relax_k),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
