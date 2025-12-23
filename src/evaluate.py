import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import JerseySequenceDataset, collate_batch
from .model import TemporalDigitNet
from .utils import set_seed
from .train import compute_metrics, select_device, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained jersey digit model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("temporal_jersey_nr_recognition_dataset_subset"),
        help="Root dataset directory.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--split-cache", type=Path, default=Path("artifacts/splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-frames-override",
        type=int,
        default=None,
        help="Optional override for max frames to evaluate (e.g., 1 for anchor-only).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string. Default runs on GPU (e.g., cuda:0). Set to cpu to force CPU.",
    )
    parser.add_argument("--keep-aspect", action="store_true", help="Keep aspect ratio by padding instead of stretching.")
    parser.add_argument("--save-scaled-dir", type=Path, default=None, help="Optional dir to save scaled images for inspection.")
    return parser.parse_args()


@torch.no_grad()
def evaluate():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = TemporalDigitNet(
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        freeze_until=cfg.get("freeze_until", 6),
        pretrained_encoder=cfg.get("pretrained_encoder", True),
        encoder_name=cfg.get("encoder", "mobilenet_v3_small"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = JerseySequenceDataset(
        root=args.data_root,
        split=args.split,
        split_path=args.split_cache,
        min_frames=3,
        max_frames=args.max_frames_override or cfg["max_frames"],
        image_size=tuple(cfg["image_size"]),
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        is_train=False,
        imagenet_stats=cfg.get("pretrained_encoder", True),
        sharpness=cfg.get("sharpness", 1.0),
        include_anchor=False,
        keep_aspect=args.keep_aspect or cfg.get("keep_aspect", False),
        save_scaled_dir=args.save_scaled_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    criterion_tens = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    criterion_ones = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    total_loss = 0.0
    total = 0
    metrics_sum: Dict[str, float] = {"tens_acc": 0.0, "ones_acc": 0.0, "number_acc": 0.0}

    for batch in loader:
        batch = to_device(batch, device)
        tens_logits, ones_logits, _ = model(batch["frames"], batch["mask"])
        loss = criterion_tens(tens_logits, batch["tens"]) + criterion_ones(ones_logits, batch["ones"])
        bs = batch["frames"].size(0)
        total_loss += loss.item() * bs
        total += bs
        m = compute_metrics(tens_logits, ones_logits, batch["tens"], batch["ones"])
        for k in metrics_sum:
            metrics_sum[k] += m[k] * bs

    metrics = {k: v / total for k, v in metrics_sum.items()}
    report = {"split": args.split, "loss": total_loss / total, **metrics}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    evaluate()
