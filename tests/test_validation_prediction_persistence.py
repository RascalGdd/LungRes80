import json

from downstream_phase.run_phase_training import (
    VideoSequentialDistributedBatchSampler,
    merge_unrelaxed_validation_predictions,
)


def test_online_sampler_keeps_whole_videos_ordered_and_unique_per_batch():
    class Dataset:
        dataset_samples = [
            {"video_id": "01", "frame_id": frame} for frame in range(5)
        ] + [
            {"video_id": "02", "frame_id": frame} for frame in range(3)
        ] + [
            {"video_id": "03", "frame_id": frame} for frame in range(4)
        ]

    rank_indices = []
    for rank in range(2):
        sampler = VideoSequentialDistributedBatchSampler(
            Dataset(), batch_size=2, num_replicas=2, rank=rank
        )
        batches = list(sampler)
        assert len(batches) == len(sampler)
        for batch in batches:
            videos = [Dataset.dataset_samples[index]["video_id"] for index in batch]
            assert len(videos) == len(set(videos))
        rank_indices.append([index for batch in batches for index in batch])

    assert sorted(rank_indices[0] + rank_indices[1]) == list(range(12))
    rank_videos = [
        {Dataset.dataset_samples[index]["video_id"] for index in indices}
        for indices in rank_indices
    ]
    assert rank_videos[0].isdisjoint(rank_videos[1])
    for indices in rank_indices:
        for video_id in rank_videos[0] | rank_videos[1]:
            frames = [
                Dataset.dataset_samples[index]["frame_id"]
                for index in indices
                if Dataset.dataset_samples[index]["video_id"] == video_id
            ]
            assert frames == sorted(frames)


def test_rank_parts_are_merged_and_all_metrics_reuse_unrelaxed(tmp_path):
    prediction_dir = tmp_path / "val_predictions" / "epoch_000"
    prediction_dir.mkdir(parents=True)
    rows_by_rank = (
        [
            {"video_id": "01", "frame_id": 0, "target": 0, "prediction": 0, "logits": [2.0, 0.0]},
            {"video_id": "01", "frame_id": 2, "target": 1, "prediction": 1, "logits": [0.0, 2.0]},
        ],
        [
            {"video_id": "01", "frame_id": 1, "target": 0, "prediction": 0, "logits": [2.0, 0.0]},
            {"video_id": "01", "frame_id": 3, "target": 1, "prediction": 1, "logits": [0.0, 2.0]},
        ],
    )
    for rank, rows in enumerate(rows_by_rank):
        part = prediction_dir / f"unrelaxed.rank{rank}.jsonl"
        part.write_text("".join(json.dumps(row) + "\n" for row in rows))

    output = merge_unrelaxed_validation_predictions(
        str(tmp_path), "epoch_000", num_tasks=2, expected_frames=4
    )

    merged = [json.loads(line) for line in open(output, encoding="utf-8")]
    assert [row["frame_id"] for row in merged] == [0, 1, 2, 3]
    assert [row["prediction"] for row in merged] == [0, 0, 1, 1]
    assert (prediction_dir / "metrics_strict.json").exists()
    assert (prediction_dir / "metrics_relaxed_k5.json").exists()
    assert not (prediction_dir / "metrics_relaxed_10s_paper.json").exists()


def test_test_split_predictions_are_saved_separately(tmp_path):
    prediction_dir = tmp_path / "test_predictions" / "epoch_000"
    prediction_dir.mkdir(parents=True)
    row = {
        "video_id": "80",
        "frame_id": 0,
        "target": 0,
        "prediction": 0,
        "logits": [2.0, 0.0],
    }
    (prediction_dir / "unrelaxed.rank0.jsonl").write_text(
        json.dumps(row) + "\n"
    )

    output = merge_unrelaxed_validation_predictions(
        str(tmp_path),
        "epoch_000",
        num_tasks=1,
        expected_frames=1,
        split="test",
    )

    assert output == str(prediction_dir / "unrelaxed.jsonl")
    assert (prediction_dir / "metrics_strict.json").exists()
    assert (prediction_dir / "metrics_relaxed_k5.json").exists()
    assert not (prediction_dir / "metrics_relaxed_10s_paper.json").exists()
