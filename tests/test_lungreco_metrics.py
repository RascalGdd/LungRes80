import numpy as np

from evaluation_lungreco import adc, collapse, evaluate, relax_prediction, segmental_f1


def test_perfect_metrics():
    target = np.asarray([0, 0, 1, 1, 2, 2])
    videos = {"01": (target.copy(), target)}
    result = evaluate(videos, relaxed=False)
    assert result["accuracy"] == 100.0
    assert result["adc"] == 0.0
    assert result["segmental_edit"] == 100.0
    assert result["segmental_f1@50"] == 100.0
    assert result["precision_per_phase"]["Fissure Dissection"] == 100.0
    assert result["recall_per_phase"]["Artery Dissection"] == 100.0
    assert result["jaccard_per_phase"]["Other Operations"] is None


def test_adc_penalizes_flicker():
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    smooth = np.asarray([0, 0, 1, 1, 1, 1, 1, 1])
    flicker = np.asarray([0, 1, 0, 1, 1, 0, 1, 0])
    assert adc(flicker, target) > adc(smooth, target)
    assert collapse(target) == [0, 1]
    assert segmental_f1(target, target, 0.5) == 100.0


def test_phase_metrics_follow_matlab_phase_then_macro_mean():
    videos = {
        "01": (np.asarray([0, 0]), np.asarray([0, 0])),
        "02": (np.asarray([0, 0]), np.asarray([1, 1])),
    }
    result = evaluate(videos, relaxed=False)
    assert result["accuracy"] == 50.0
    assert result["precision"] == 100.0
    assert result["recall"] == 50.0
    assert result["jaccard"] == 50.0


def test_relax_matches_lungres80_adjacent_phase_k5_rule():
    target = np.asarray([0] * 8 + [1] * 8)
    prediction = target.copy()
    prediction[6] = 1  # adjacent next phase near the boundary: accepted
    prediction[7] = 6  # unrelated phase near the boundary: remains wrong
    relaxed = relax_prediction(prediction, target, radius=5)
    assert relaxed[6] == target[6]
    assert relaxed[7] == 6


def test_segmental_f1_is_split_level_micro_not_video_mean():
    videos = {
        "01": (np.asarray([0, 1, 2]), np.asarray([0, 1, 2])),
        "02": (np.asarray([1]), np.asarray([0])),
    }
    result = evaluate(videos, relaxed=False)
    assert result["segmental_f1@50"] == 75.0
    assert result["segmental_precision@50"] == 75.0
    assert result["segmental_recall@50"] == 75.0
