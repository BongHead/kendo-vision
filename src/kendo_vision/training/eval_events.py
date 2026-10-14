"""Scores predicted events against labelled ones."""
from __future__ import annotations

from collections import Counter

from ..io.sink import read_jsonl
from ..constants import TARGETS


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total} = {count / total:.2f}" if total else f"{count}/{total}"


def evaluate(predicted_path: str, truth_path: str, tolerance: float = 0.25) -> None:
    """Pairs events within `tolerance` seconds, closest first, each used once."""
    predicted = read_jsonl(predicted_path)
    # labels marked "none" are rejected candidates, not strikes to be found
    truth = [event for event in read_jsonl(truth_path) if (event.get("target") or "none") != "none"]

    # every predicted/true pair that is close enough in time, nearest pairs first
    close_pairs = []
    for predicted_index, prediction in enumerate(predicted):
        for truth_index, actual in enumerate(truth):
            time_apart = abs(prediction["time"] - actual["time"])
            if time_apart <= tolerance:
                close_pairs.append((time_apart, predicted_index, truth_index))
    close_pairs.sort()
    # take pairs greedily, so each event is matched at most once and to its nearest free partner
    used_predicted, used_truth, matches = set(), set(), []
    for _, predicted_index, truth_index in close_pairs:
        if predicted_index not in used_predicted and truth_index not in used_truth:
            used_predicted.add(predicted_index)
            used_truth.add(truth_index)
            matches.append((predicted[predicted_index], truth[truth_index]))

    target_hits = sum(prediction["target"] == actual["target"] for prediction, actual in matches)
    attacker_hits = sum(prediction["attacker"] == actual["attacker"] for prediction, actual in matches)
    print(f"Strikes found:    precision {_ratio(len(matches), len(predicted))}, "
          f"recall {_ratio(len(matches), len(truth))}")
    print(f"Target correct:   {_ratio(target_hits, len(matches))}")
    print(f"Attacker correct: {_ratio(attacker_hits, len(matches))}")

    confusion = Counter((actual["target"], prediction["target"]) for prediction, actual in matches)
    print("Target confusion (rows = truth, columns = predicted):")
    print("        " + "".join(f"{target:>7}" for target in TARGETS))
    for actual_target in TARGETS:
        counts = "".join(f"{confusion[(actual_target, predicted_target)]:>7}" for predicted_target in TARGETS)
        print(f"{actual_target:>7} {counts}")

    # ippon is only scored where the label says yes or no
    judged = [(prediction, actual) for prediction, actual in matches if actual.get("ippon") is not None]
    if judged:
        correct_ippon = sum(prediction.get("verdict") == "ippon" and actual["ippon"] for prediction, actual in judged)
        called_ippon = sum(prediction.get("verdict") == "ippon" for prediction in predicted)
        actual_ippon = sum(bool(event.get("ippon")) for event in truth)
        print(f"Ippon:            precision {_ratio(correct_ippon, called_ippon)}, "
              f"recall {_ratio(correct_ippon, actual_ippon)}")
