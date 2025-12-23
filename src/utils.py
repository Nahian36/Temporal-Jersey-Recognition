import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


TENS_BLANK = 10  # extra class to encode single-digit numbers


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SequenceRecord:
    frames: List[Path]
    label: str
    anchor: Optional[Path] = None


def label_to_digits(label: str) -> Tuple[int, int]:
    """Map jersey label string to digit tuple (tens, ones)."""
    label = label.strip()
    if len(label) == 1:
        return TENS_BLANK, int(label)
    if len(label) == 2:
        return int(label[0]), int(label[1])
    raise ValueError(f"Unexpected jersey label: {label}")


def digits_to_label(tens: int, ones: int) -> str:
    if tens == TENS_BLANK:
        return str(ones)
    return f"{tens}{ones}"


def save_split(path: Path, split: Dict[str, List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(split, f, indent=2)


def load_split(path: Path) -> Optional[Dict[str, List[int]]]:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def stratified_split(
    records: Sequence[SequenceRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    """Return indices for train/val/test splits with per-class stratification."""
    rng = random.Random(seed)
    by_class: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        by_class.setdefault(rec.label, []).append(idx)

    split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        val_idxs = idxs[:n_val]
        test_idxs = idxs[n_val : n_val + n_test]
        train_idxs = idxs[n_val + n_test :]
        split["train"].extend(train_idxs)
        split["val"].extend(val_idxs)
        split["test"].extend(test_idxs)
    return split


def compute_digit_weights(records: Sequence[SequenceRecord], indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tens_counts = torch.zeros(11, dtype=torch.float32)
    ones_counts = torch.zeros(10, dtype=torch.float32)
    for idx in indices:
        tens, ones = label_to_digits(records[idx].label)
        tens_counts[tens] += 1
        ones_counts[ones] += 1
    eps = 1e-6
    tens_median = tens_counts[tens_counts > 0].median() if (tens_counts > 0).any() else torch.tensor(1.0)
    ones_median = ones_counts[ones_counts > 0].median() if (ones_counts > 0).any() else torch.tensor(1.0)
    tens_w = tens_median / (tens_counts + eps)
    ones_w = ones_median / (ones_counts + eps)
    return tens_w, ones_w


def clamp_weights(t: torch.Tensor, max_value: float) -> torch.Tensor:
    return torch.clamp(t, max=max_value)
