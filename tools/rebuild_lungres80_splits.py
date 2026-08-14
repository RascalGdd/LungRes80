#!/usr/bin/env python3
"""Rebuild LungRes80 split pickles from the experiment video IDs."""

from __future__ import annotations

import argparse
import pickle
import shutil
from datetime import datetime
from pathlib import Path


SPLIT_IDS = {
    "train": {
        "50", "64", "24", "51", "31", "56", "09", "47", "58", "40",
        "16", "73", "60", "39", "43", "63", "26", "57", "77", "02",
        "71", "54", "19", "68", "20", "21", "44", "48", "34", "62",
        "33", "08", "66", "03", "17", "37", "23", "41", "25", "06",
        "07", "61", "74", "10", "53", "22", "67", "59", "11", "49",
    },
    "val": {
        "30", "28", "72", "79", "05", "55", "12", "70", "14", "18",
        "29", "32", "36", "04", "15",
    },
    "test": {
        "01", "52", "75", "38", "76", "80", "27", "35", "45", "42",
        "46", "13", "78", "69", "65",
    },
}

OUTPUT_NAMES = {
    "train": "1fpstrain.pickle",
    "val": "1fpsval.pickle",
    "test": "1fpstest.pickle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="LungRes80 dataset root")
    parser.add_argument(
        "--no-backup", action="store_true", help="Do not back up existing pickles"
    )
    return parser.parse_args()


def validate_ids() -> None:
    expected = {f"{idx:02d}" for idx in range(1, 81)}
    union = set().union(*SPLIT_IDS.values())
    duplicates = {
        video_id
        for video_id in union
        if sum(video_id in ids for ids in SPLIT_IDS.values()) != 1
    }
    if union != expected or duplicates:
        raise ValueError(
            f"Invalid split IDs: missing={sorted(expected - union)}, "
            f"extra={sorted(union - expected)}, duplicates={sorted(duplicates)}"
        )


def read_annotation(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().strip().split()
        if header != ["Frame", "Phase"]:
            raise ValueError(f"Unexpected header in {path}: {header}")
        phases: list[int] = []
        for expected_frame, line in enumerate(handle):
            fields = line.strip().split()
            if len(fields) != 2:
                raise ValueError(f"Malformed annotation in {path}: {line!r}")
            frame_id, phase = map(int, fields)
            if frame_id != expected_frame:
                raise ValueError(
                    f"Non-contiguous frames in {path}: {frame_id} != {expected_frame}"
                )
            if phase not in range(7):
                raise ValueError(f"Invalid phase {phase} in {path}")
            phases.append(phase)
    return phases


def build_split(root: Path, split: str) -> dict[str, list[dict[str, int | str]]]:
    result: dict[str, list[dict[str, int | str]]] = {}
    unique_id = 0
    for video_id in sorted(SPLIT_IDS[split], key=int):
        annotation_path = root / "phase_annotations" / f"{video_id}.txt"
        frame_dir = root / "frames" / video_id
        phases = read_annotation(annotation_path)
        jpg_count = sum(1 for path in frame_dir.iterdir() if path.suffix.lower() == ".jpg")
        if jpg_count != len(phases):
            raise ValueError(
                f"Frame/annotation mismatch for {video_id}: {jpg_count} != {len(phases)}"
            )
        rows = []
        for frame_id, phase in enumerate(phases):
            frame_path = frame_dir / f"{frame_id:06d}.jpg"
            if not frame_path.is_file():
                raise FileNotFoundError(frame_path)
            rows.append(
                {
                    "unique_id": unique_id,
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "frames": len(phases),
                    "phase_gt": phase,
                    "fps": 1,
                }
            )
            unique_id += 1
        result[video_id] = rows
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    validate_ids()
    outputs = {
        split: root / "labels" / split / filename
        for split, filename in OUTPUT_NAMES.items()
    }

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = root / "labels" / f"backup_before_random_split_{stamp}"
        for split, output in outputs.items():
            if output.exists():
                destination = backup_root / split / output.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output, destination)
        print(f"backup={backup_root}")

    for split, output in outputs.items():
        data = build_split(root, split)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(data, handle, protocol=4)
        temporary.replace(output)
        rows = sum(len(items) for items in data.values())
        print(
            f"{split}: videos={len(data)} rows={rows} "
            f"ids={','.join(data.keys())} output={output}"
        )


if __name__ == "__main__":
    main()
