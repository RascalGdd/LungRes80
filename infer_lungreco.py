#!/usr/bin/env python3
"""Sequential online LungReco inference that writes metric-ready JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from downstream_phase.datasets_phase import build_dataset
from downstream_phase.run_phase_training import VideoSequentialDistributedBatchSampler
from model.lungreco import lungreco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llava_model_path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--pretrained_path", default="")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gaussian_history_length", type=int, default=64)
    parser.add_argument("--gaussian_sigma", type=float, default=3.0)
    parser.add_argument(
        "--history_source",
        choices=("predicted", "gt"),
        default="predicted",
        help="Use autoregressive predicted transitions or label-derived GT transitions.",
    )
    parser.add_argument(
        "--use_text_cost_fusion",
        action="store_true",
        help="Enable the first fusion: include the LLaVA history token in CoST attention.",
    )
    parser.add_argument(
        "--video_ids",
        nargs="*",
        default=None,
        help="Optional video-ID subset; useful for exact video-wise multi-GPU sharding",
    )
    return parser.parse_args()


def dataset_args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_set="LungRes80",
        data_path=str(root),
        data_strategy="online",
        output_mode="key_frame",
        cut_black=False,
        num_frames=16,
        sampling_rate=8,
        input_size=224,
        short_side_size=224,
        nb_classes=7,
        reprob=0.0,
        paper_augmentation=True,
        aa="",
        train_interpolation="bicubic",
        remode="pixel",
        recount=1,
    )


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    for key in ("model", "module"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    checkpoint = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in checkpoint.items()
    }
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def main() -> None:
    args = parse_args()
    ds_args = dataset_args(args.data_root)
    dataset, _ = build_dataset(
        is_train=False,
        test_mode=args.split == "test",
        fps="1fps",
        args=ds_args,
    )
    if args.video_ids:
        selected = set(args.video_ids)
        indices = [
            index
            for index, sample in enumerate(dataset.dataset_samples)
            if str(sample["video_id"]) in selected
        ]
        found = {str(dataset.dataset_samples[index]["video_id"]) for index in indices}
        missing = selected - found
        if missing:
            raise ValueError(f"Unknown video IDs for {args.split}: {sorted(missing)}")
        dataset = Subset(dataset, indices)
    sampler_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if isinstance(dataset, Subset):
        # A video-ID subset contains complete videos, so rebuild the lightweight
        # view needed by the chronological batch sampler.
        class DatasetView:
            dataset_samples = [sampler_dataset.dataset_samples[i] for i in dataset.indices]

        sampler_source = DatasetView()
    else:
        sampler_source = dataset
    batch_sampler = VideoSequentialDistributedBatchSampler(
        sampler_source, args.batch_size, num_replicas=1, rank=0
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = lungreco(
        pretrained=bool(args.pretrained_path),
        pretrain_path=args.pretrained_path or None,
        num_classes=7,
        all_frames=16,
        llava_model_path=args.llava_model_path,
        gaussian_history_length=args.gaussian_history_length,
        gaussian_sigma=args.gaussian_sigma,
        use_text_cost_fusion=args.use_text_cost_fusion,
    )
    load_checkpoint(model, args.checkpoint)
    model.cuda().eval()
    model.reset_stream_state()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle, torch.no_grad():
        for videos, targets, identifiers, metadata in loader:
            videos = videos.cuda(non_blocking=True)
            video_ids = [identifier.split("_")[1] for identifier in identifiers]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.history_source == "gt":
                    padded_histories = metadata["gt_recent_phases"]
                    history_counts = metadata["gt_recent_phase_count"]
                    histories = [
                        row[: int(count)].tolist()
                        for row, count in zip(padded_histories, history_counts)
                    ]
                    raw_logits = model(
                        videos, recent_phase_histories=histories
                    )
                    result = model._update_stream_state(raw_logits, video_ids)
                    result["raw_logits"] = raw_logits
                else:
                    result = model.predict_online(videos, video_ids)
            raw_logits = result["raw_logits"].float().cpu()
            raw_probabilities = result["raw_probabilities"].cpu()
            smoothed_probabilities = result["smoothed_probabilities"].cpu()
            predictions = result["smoothed_predictions"].cpu().tolist()
            for identifier, prediction, target, logits, raw_probs, smooth_probs in zip(
                identifiers,
                predictions,
                targets.tolist(),
                raw_logits.tolist(),
                raw_probabilities.tolist(),
                smoothed_probabilities.tolist(),
            ):
                _, video_id, frame_id = identifier.split("_")
                handle.write(
                    json.dumps(
                        {
                            "video_id": video_id,
                            "frame_id": int(frame_id),
                            "prediction": int(prediction),
                            "raw_prediction": int(torch.tensor(logits).argmax().item()),
                            "target": int(target),
                            "logits": logits,
                            "raw_probabilities": raw_probs,
                            "smoothed_probabilities": smooth_probs,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
