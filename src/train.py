import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    class _TQDMStub:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            return None

    def tqdm(iterable, **kwargs):  # type: ignore
        return _TQDMStub(iterable, **kwargs)

from .dataset import JerseySequenceDataset, collate_batch
from .model import TemporalDigitNet
from .utils import clamp_weights, compute_digit_weights, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal jersey digit recognizer.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("temporal_jersey_nr_recognition_dataset_subset"),
        help="Root folder with class-labelled sequence directories.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-until-layer", type=int, default=6, help="Freeze early encoder blocks for stability.")
    parser.add_argument("--no-pretrained-encoder", action="store_true", help="Disable ImageNet pretraining on the encoder.")
    parser.add_argument(
        "--encoder",
        type=str,
        default="mobilenet_v3_small",
        help="Encoder backbone: mobilenet_v3_small|mobilenet_v3_large|efficientnet_lite0 (alias: efficientnet_b0).",
    )
    parser.add_argument("--frame-drop-prob", type=float, default=0.02, help="Random frame drop probability during training.")
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable digit-balanced loss weighting.")
    parser.add_argument("--max-class-weight", type=float, default=3.0, help="Clamp for digit class weights to avoid instability.")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Epochs for linear LR warmup before cosine schedule.")
    parser.add_argument("--sharpness", type=float, default=1.1, help="Mild sharpness adjustment for upsampled low-res inputs (1.0 disables).")
    parser.add_argument("--no-anchor", action="store_true", help="Disable using anchor.jpg during training even if present.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (recommended on CUDA).")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pinned memory for faster host->GPU transfer.")
    parser.add_argument("--keep-aspect", action="store_true", help="Keep aspect ratio by padding instead of stretching.")
    parser.add_argument("--save-scaled-dir", type=Path, default=None, help="Optional dir to save scaled images for inspection.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for model weights (0 disables EMA).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional limit for quick iterations.")
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string. Default runs on GPU (e.g., cuda:0). Set to cpu to force CPU.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience based on val loss.")
    parser.add_argument(
        "--split-cache",
        type=Path,
        default=Path("artifacts/splits.json"),
        help="Path to persist dataset split indices.",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    normalized = device_arg.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            return device
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized.startswith("gpu"):
        suffix = device_arg.split(":", 1)[1] if ":" in device_arg else "0"
        device_str = f"cuda:{suffix}"
    elif normalized.startswith("cuda"):
        device_str = device_arg if ":" in device_arg else "cuda:0"
    elif normalized == "cpu":
        device_str = "cpu"
    else:
        device_str = device_arg

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU.")
            return torch.device("cpu")
        torch.backends.cudnn.benchmark = True
    return torch.device(device_str)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        "frames": batch["frames"].to(device),
        "mask": batch["mask"].to(device),
        "tens": batch["tens"].to(device),
        "ones": batch["ones"].to(device),
        "labels": batch["labels"],
    }


def compute_metrics(tens_logits, ones_logits, tens_target, ones_target) -> Dict[str, float]:
    tens_pred = tens_logits.argmax(dim=1)
    ones_pred = ones_logits.argmax(dim=1)
    full_correct = (tens_pred == tens_target) & (ones_pred == ones_target)
    return {
        "tens_acc": (tens_pred == tens_target).float().mean().item(),
        "ones_acc": (ones_pred == ones_target).float().mean().item(),
        "number_acc": full_correct.float().mean().item(),
    }


@torch.no_grad()
def run_eval(
    model: TemporalDigitNet,
    loader: DataLoader,
    device: torch.device,
    criterion_tens: nn.Module,
    criterion_ones: nn.Module,
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total = 0
    metrics_sum = {"tens_acc": 0.0, "ones_acc": 0.0, "number_acc": 0.0}
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(enabled=use_amp):
            tens_logits, ones_logits, _ = model(batch["frames"], batch["mask"])
            loss = criterion_tens(tens_logits, batch["tens"]) + criterion_ones(ones_logits, batch["ones"])
        bs = batch["frames"].size(0)
        total_loss += loss.item() * bs
        total += bs
        m = compute_metrics(tens_logits, ones_logits, batch["tens"], batch["ones"])
        for k in metrics_sum:
            metrics_sum[k] += m[k] * bs
    if total == 0:
        return 0.0, {k: 0.0 for k in metrics_sum}
    return total_loss / total, {k: v / total for k, v in metrics_sum.items()}


def train():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    image_size = (args.image_height, args.image_width)
    use_imagenet_stats = not args.no_pretrained_encoder

    train_ds = JerseySequenceDataset(
        root=args.data_root,
        split="train",
        split_path=args.split_cache,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        image_size=image_size,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        is_train=True,
        frame_drop_prob=args.frame_drop_prob,
        imagenet_stats=use_imagenet_stats,
        sharpness=args.sharpness,
        include_anchor=not args.no_anchor,
        keep_aspect=args.keep_aspect,
        save_scaled_dir=args.save_scaled_dir,
    )
    val_ds = JerseySequenceDataset(
        root=args.data_root,
        split="val",
        split_path=args.split_cache,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        image_size=image_size,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        is_train=False,
        imagenet_stats=use_imagenet_stats,
        sharpness=1.0,
        include_anchor=False,
        keep_aspect=args.keep_aspect,
        save_scaled_dir=args.save_scaled_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
    )

    model = TemporalDigitNet(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_until=args.freeze_until_layer,
        pretrained_encoder=not args.no_pretrained_encoder,
        encoder_name=args.encoder,
    ).to(device)

    tens_w, ones_w = (None, None)
    if not args.no_class_weights:
        tens_w, ones_w = compute_digit_weights(train_ds.records, train_ds.indices)
        tens_w = clamp_weights(tens_w, args.max_class_weight).to(device)
        ones_w = clamp_weights(ones_w, args.max_class_weight).to(device)

    criterion_tens = nn.CrossEntropyLoss(weight=tens_w, label_smoothing=args.label_smoothing)
    criterion_ones = nn.CrossEntropyLoss(weight=ones_w, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = GradScaler(enabled=use_amp)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    run_dir = args.save_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.json"
    best_path = args.save_dir / "best_model.pt"

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        total_loss = 0.0
        total = 0
        metrics_sum = {"tens_acc": 0.0, "ones_acc": 0.0, "number_acc": 0.0}
        for batch in pbar:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                tens_logits, ones_logits, _ = model(batch["frames"], batch["mask"])
                loss = criterion_tens(tens_logits, batch["tens"]) + criterion_ones(ones_logits, batch["ones"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            bs = batch["frames"].size(0)
            total_loss += loss.item() * bs
            total += bs
            m = compute_metrics(tens_logits, ones_logits, batch["tens"], batch["ones"])
            for k in metrics_sum:
                metrics_sum[k] += m[k] * bs
            pbar.set_postfix({"loss": loss.item(), "num_acc": m["number_acc"]})

        scheduler.step()

        train_loss = total_loss / total
        train_metrics = {k: v / total for k, v in metrics_sum.items()}

        # Evaluate using EMA weights if enabled
        if ema is not None:
            ema.apply_to(model)
        val_loss, val_metrics = run_eval(model, val_loader, device, criterion_tens, criterion_ones, use_amp)
        if ema is not None:
            ema.restore(model)

        history["train"].append({"loss": train_loss, **train_metrics})
        history["val"].append({"loss": val_loss, **val_metrics})

        print(
            f"Epoch {epoch}: train loss {train_loss:.4f} num_acc {train_metrics['number_acc']:.3f} | "
            f"val loss {val_loss:.4f} val num_acc {val_metrics['number_acc']:.3f}"
        )

        with history_path.open("w") as f:
            json.dump(history, f, indent=2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path.parent.mkdir(parents=True, exist_ok=True)
            state_to_save = ema.shadow if ema is not None else model.state_dict()
            torch.save(
                {
                    "model_state": state_to_save,
                    "config": {
                        "embed_dim": args.embed_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "image_size": image_size,
                        "max_frames": args.max_frames,
                        "freeze_until": args.freeze_until_layer,
                        "pretrained_encoder": not args.no_pretrained_encoder,
                        "encoder": args.encoder,
                        "label_smoothing": args.label_smoothing,
                        "frame_drop_prob": args.frame_drop_prob,
                        "sharpness": args.sharpness,
                        "use_anchor": not args.no_anchor,
                        "keep_aspect": args.keep_aspect,
                    },
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    _maybe_plot_history(history, run_dir / "curves.png")
    print(f"Training finished. Best val loss: {best_val_loss:.4f}. Model saved to {best_path}")


def _maybe_plot_history(history: Dict, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"Skipping curve plot due to error: {exc}")
        return
    try:
        epochs = range(1, len(history["train"]) + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, [h["loss"] for h in history["train"]], label="train")
        plt.plot(epochs, [h["loss"] for h in history["val"]], label="val")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, [h["number_acc"] for h in history["train"]], label="train num acc")
        plt.plot(epochs, [h["number_acc"] for h in history["val"]], label="val num acc")
        plt.title("Number accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
    except Exception as exc:
        print(f"Skipping curve plot due to error: {exc}")


def _build_scheduler(optimizer: optim.Optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int):
        e = epoch
        if warmup_epochs > 0 and e < warmup_epochs:
            return (e + 1) / float(warmup_epochs)
        progress = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


if __name__ == "__main__":
    train()
