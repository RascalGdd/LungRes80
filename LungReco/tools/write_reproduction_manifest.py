#!/usr/bin/env python3
"""Write a machine-verifiable manifest for a LungReco experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CODE_FILES = (
    "model/lungreco.py",
    "datasets/phase/LungRes80_phase.py",
    "downstream_phase/engine_for_phase.py",
    "downstream_phase/run_phase_training.py",
    "utils.py",
    "infer_lungreco.py",
    "evaluation_lungreco.py",
    "scripts/train_lungreco_lungres80.sh",
    "scripts/evaluate_best_lungreco_lungres80.sh",
    "tools/rebuild_lungres80_splits.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, include_hash: bool = True) -> dict:
    record = {"path": str(path.resolve()), "bytes": path.stat().st_size}
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def split_record(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    rows = [row for video_rows in data.values() for row in video_rows]
    return {
        **file_record(path),
        "videos": len(data),
        "frames": len(rows),
        "video_ids": list(data),
        "phase_counts": dict(sorted(Counter(row["phase_gt"] for row in rows).items())),
    }


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--llava-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    split_paths = {
        "train": args.dataset_root / "labels/train/1fpstrain.pickle",
        "val": args.dataset_root / "labels/val/1fpsval.pickle",
        "test": args.dataset_root / "labels/test/1fpstest.pickle",
    }
    llava_files = []
    for name in ("config.json", "tokenizer.model", "model.safetensors.index.json"):
        path = args.llava_path / name
        if path.is_file():
            llava_files.append(file_record(path))
    for path in sorted(args.llava_path.glob("*.safetensors")):
        llava_files.append(file_record(path, include_hash=False))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "timm", "transformers", "deepspeed", "xlstm")
        },
        "source_commit": git_value(root, "rev-parse", "HEAD"),
        "git_status": git_value(root, "status", "--short"),
        "code": {name: file_record(root / name) for name in CODE_FILES},
        "dataset": {name: split_record(path) for name, path in split_paths.items()},
        "pretrained": file_record(args.pretrained),
        "llava_snapshot": {
            "path": str(args.llava_path.resolve()),
            "files": llava_files,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
