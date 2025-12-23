import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils import (
    SequenceRecord,
    TENS_BLANK,
    digits_to_label,
    label_to_digits,
    load_split,
    save_split,
    stratified_split,
)


def _discover_sequences(root: Path, min_frames: int = 3) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for seq_dir in sorted(label_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            frames = sorted(
                [p for p in seq_dir.rglob("*.jpg") if p.name.lower() != "anchor.jpg"]
            )
            anchor = seq_dir / "anchor.jpg"
            anchor = anchor if anchor.exists() else None
            if len(frames) >= min_frames:
                records.append(SequenceRecord(frames=frames, label=label, anchor=anchor))
    return records


def default_transform(
    image_size: Tuple[int, int],
    imagenet_stats: bool = True,
    sharpness: float = 1.0,
    keep_aspect: bool = False,
    save_scaled_dir: Optional[Path] = None,
    label: Optional[str] = None,
    idx: Optional[int] = None,
) -> Callable:
    """High-quality resize with optional aspect-ratio preservation and saving scaled images."""
    h, w = image_size
    mean = [0.485, 0.456, 0.406] if imagenet_stats else [0.5, 0.5, 0.5]
    std = [0.229, 0.224, 0.225] if imagenet_stats else [0.25, 0.25, 0.25]

    def resize_and_pad(img: Image.Image) -> Image.Image:
        if not keep_aspect:
            return img.resize((w, h), resample=Image.BICUBIC)
        orig_w, orig_h = img.size
        scale = min(w / orig_w, h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        pad_w = w - new_w
        pad_h = h - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(img_resized, padding, fill=(128, 128, 128))

    def save_if_needed(img: Image.Image, frame_idx: int):
        if save_scaled_dir is None:
            return
        dir_path = save_scaled_dir / (label or "unknown")
        dir_path.mkdir(parents=True, exist_ok=True)
        fname = f"{idx if idx is not None else 0}_{frame_idx}.jpg"
        img.save(dir_path / fname)

    def _transform(img: Image.Image, frame_idx: int):
        img = resize_and_pad(img)
        if sharpness != 1.0:
            img = transforms.functional.adjust_sharpness(img, sharpness)
        save_if_needed(img, frame_idx)
        img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(img)
        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=mean, std=std)(tensor)
        return tensor

    return _transform


class JerseySequenceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        split_path: Optional[Path],
        min_frames: int = 3,
        max_frames: int = 16,
        image_size: Tuple[int, int] = (96, 64),
        limit_per_class: Optional[int] = None,
        seed: int = 42,
        transform: Optional[Callable] = None,
        is_train: bool = False,
        frame_drop_prob: float = 0.0,
        imagenet_stats: bool = True,
        sharpness: float = 1.0,
        include_anchor: bool = True,
        keep_aspect: bool = False,
        save_scaled_dir: Optional[Path] = None,
    ) -> None:
        self.root = root
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.image_size = image_size
        self.keep_aspect = keep_aspect
        self.save_scaled_dir = save_scaled_dir
        self._transform_factory = transform or default_transform
        self._transform_args = {
            "image_size": image_size,
            "imagenet_stats": imagenet_stats,
            "sharpness": sharpness,
            "keep_aspect": keep_aspect,
            "save_scaled_dir": save_scaled_dir,
        }
        self.is_train = is_train
        self.frame_drop_prob = frame_drop_prob if is_train else 0.0
        self.include_anchor = include_anchor if is_train else False
        self.records = _discover_sequences(root, min_frames=min_frames)

        if not self.records:
            raise ValueError(f"No sequences found under {root}")

        loaded_split = load_split(split_path) if split_path else None
        if loaded_split is None:
            split_indices = stratified_split(
                self.records, val_ratio=0.1, test_ratio=0.1, seed=seed
            )
            if split_path:
                save_split(split_path, split_indices)
        else:
            split_indices = loaded_split

        if split not in split_indices:
            raise ValueError(f"Split '{split}' not in {list(split_indices.keys())}")

        selected_indices = split_indices[split]
        if limit_per_class is not None:
            selected_indices = self._limit_by_class(selected_indices, limit_per_class)
        self.indices = selected_indices

    def _limit_by_class(self, indices: Sequence[int], limit: int) -> List[int]:
        per_class: Dict[str, List[int]] = {}
        for idx in indices:
            label = self.records[idx].label
            per_class.setdefault(label, []).append(idx)
        trimmed: List[int] = []
        for label, idxs in per_class.items():
            trimmed.extend(idxs[:limit])
        return trimmed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        rec = self.records[self.indices[idx]]
        frames = rec.frames
        # Inject anchor for training only (as best single view) without exceeding max_frames.
        if self.include_anchor and rec.anchor is not None:
            base = [rec.anchor] + frames
            if len(base) <= self.max_frames:
                sampled = base
            else:
                remaining = self.max_frames - 1
                idxs = np.linspace(0, len(frames) - 1, num=remaining, dtype=int)
                sampled = [rec.anchor] + [frames[i] for i in idxs]
        else:
            if len(frames) <= self.max_frames:
                sampled = frames
            else:
                idxs = np.linspace(0, len(frames) - 1, num=self.max_frames, dtype=int)
                sampled = [frames[i] for i in idxs]
        if self.is_train and self.frame_drop_prob > 0.0:
            kept = []
            for p in sampled:
                # keep the first frame (often anchor) to avoid dropping the clearest view
                if len(kept) == 0 or random.random() > self.frame_drop_prob:
                    kept.append(p)
            sampled = kept

        transform = self._transform_factory(
            **self._transform_args, label=rec.label, idx=self.indices[idx]
        )
        images = [
            transform(Image.open(p).convert("RGB"), frame_idx=i) for i, p in enumerate(sampled)
        ]
        seq_len = len(images)

        if seq_len < self.max_frames:
            pad = torch.zeros(
                (self.max_frames - seq_len, *images[0].shape), dtype=images[0].dtype
            )
            images = images + [pad[i] for i in range(pad.shape[0])]

        tensor = torch.stack(images, dim=0)  # (T, C, H, W)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        mask[:seq_len] = True
        tens, ones = label_to_digits(rec.label)
        return {
            "frames": tensor,
            "mask": mask,
            "tens": torch.tensor(tens, dtype=torch.long),
            "ones": torch.tensor(ones, dtype=torch.long),
            "label": rec.label,
        }


def collate_batch(batch: List[Dict]):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    tens = torch.stack([b["tens"] for b in batch], dim=0)
    ones = torch.stack([b["ones"] for b in batch], dim=0)
    labels = [b["label"] for b in batch]
    return {"frames": frames, "mask": mask, "tens": tens, "ones": ones, "labels": labels}
