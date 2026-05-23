#!/usr/bin/env python3
"""Train MiniSSRSegNetV3 phase-2 experiments on preprocessed ACDC data."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

from acdc_dataset import ACDCSSRSliceDataset, load_or_create_split
from ssr_blocks import boundary_map_from_mask, build_radial_frequency_masks
from ssr_model import MiniSSRSegNetV3


CLASS_NAMES = ["BG", "RV", "MYO", "LV"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSRBlockV3 ACDC phase-2 training")
    parser.add_argument("--config", default="test/configs/ssr_v3_acdc.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_config_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("run_name", "ssr_v3_acdc_debug")
    cfg.setdefault("seed", 42)
    cfg.setdefault("data_root", "preprocessed_data/ACDC")
    cfg.setdefault("output_root", "test/outputs")
    cfg.setdefault("input_mode", "2d")
    cfg.setdefault("image_size", 128)
    cfg.setdefault("foreground_only", True)
    cfg.setdefault("batch_size", 4)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("epochs", 80)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("base_channels", 32)
    cfg.setdefault("num_classes", 4)
    cfg.setdefault("num_bands", 4)
    cfg.setdefault("grad_clip", 3.0)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("split_manifest", "splits/acdc_patient_split_seed42.json")
    cfg.setdefault("variant", "default")

    loss_weights = cfg.setdefault("loss_weights", {})
    loss_weights.setdefault("boundary_bce", 0.50)
    loss_weights.setdefault("boundary_dice", 0.30)
    loss_weights.setdefault("boundary_frequency", 0.20)
    loss_weights.setdefault("tv", 0.05)
    loss_weights.setdefault("gate_reg", 0.03)
    loss_weights.setdefault("hf_ratio", 0.005)

    ssr = cfg.setdefault("ssr", {})
    ssr.setdefault("update_budget", 1.5)
    ssr.setdefault("min_update", 0.08)
    ssr.setdefault("noise_strength", 0.04)
    ssr.setdefault("retain_floor", [0.15, 0.18, 0.22, 0.28])
    ssr.setdefault("suppress_min", [0.00, 0.02, 0.03, 0.03])
    ssr.setdefault("suppress_max", [0.05, 0.15, 0.25, 0.25])
    ssr.setdefault("update_target", [0.30, 0.30, 0.25, 0.15])
    ssr.setdefault("noise_aware_suppress", True)
    ssr.setdefault("use_bounded_gamma", True)
    ssr.setdefault("gamma_max", 0.25)
    ssr.setdefault("gamma_init", -2.0)
    ssr.setdefault("residual_gate_type", "se_update")
    ssr.setdefault("se_reduction", 4)
    ssr.setdefault("geometry_refine", "none")
    ssr.setdefault("large_kernel_size", 7)
    ssr.setdefault("dcnv4_group", 2)
    ssr.setdefault("use_hf_ratio_penalty", True)
    ssr.setdefault("hf_ratio_threshold", 4.0)
    return cfg


def apply_variant_and_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = apply_config_defaults(cfg)
    requested_variant = args.variant
    if requested_variant:
        variants = cfg.get("variants", {})
        if requested_variant not in variants:
            raise ValueError(f"Unknown variant {requested_variant!r}. Available: {sorted(variants)}")
        cfg = deep_update(cfg, variants[requested_variant])
        cfg["variant"] = requested_variant
        if args.run_name is None:
            cfg["run_name"] = f"{cfg['run_name']}_{requested_variant}"

    for key in (
        "epochs",
        "batch_size",
        "image_size",
        "max_slices",
        "run_name",
        "device",
        "data_root",
        "output_root",
        "num_workers",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return apply_config_defaults(cfg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def make_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_cases, val_cases, split_manifest = load_or_create_split(
        cfg["data_root"],
        seed=int(cfg["seed"]),
        train_fraction=0.8,
        split_manifest=cfg.get("split_manifest", "splits/acdc_patient_split_seed42.json"),
    )
    train_ds = ACDCSSRSliceDataset(
        cfg["data_root"],
        case_ids=train_cases,
        input_mode=cfg["input_mode"],
        image_size=int(cfg["image_size"]),
        foreground_only=bool(cfg["foreground_only"]),
        max_slices=cfg.get("max_slices"),
        seed=int(cfg["seed"]),
    )
    val_ds = ACDCSSRSliceDataset(
        cfg["data_root"],
        case_ids=val_cases,
        input_mode=cfg["input_mode"],
        image_size=int(cfg["image_size"]),
        foreground_only=bool(cfg["foreground_only"]),
        max_slices=cfg.get("max_slices"),
        seed=int(cfg["seed"]) + 1,
    )

    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, {
        "train_cases": train_cases,
        "val_cases": val_cases,
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
        "split_manifest": split_manifest,
    }


def build_model(cfg: dict[str, Any]) -> MiniSSRSegNetV3:
    in_channels = 5 if str(cfg["input_mode"]).lower() == "25d" else 1
    ssr_cfg = dict(cfg.get("ssr", {}))
    ssr_cfg.setdefault("num_bands", int(cfg["num_bands"]))
    return MiniSSRSegNetV3(
        in_channels=in_channels,
        base_channels=int(cfg["base_channels"]),
        num_classes=int(cfg["num_classes"]),
        num_bands=int(cfg["num_bands"]),
        ssr=ssr_cfg,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def foreground_dice_loss(logits: Tensor, target: Tensor, num_classes: int) -> Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice[1:].mean() if num_classes > 1 else 1.0 - dice.mean()


def boundary_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    pred = torch.sigmoid(logits)
    target = target.float()
    inter = (pred * target).sum(dim=(0, 2, 3))
    denom = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice.mean()


def boundary_frequency_loss(logits: Tensor, target: Tensor, num_bands: int = 4) -> Tensor:
    pred = torch.sigmoid(logits).float()
    target = target.float()
    _, _, H, W = pred.shape
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    masks = build_radial_frequency_masks(H, W, num_bands, pred.device)
    high_mask = masks[max(num_bands - 2, 0):].sum(dim=0).clamp(0, 1).view(1, 1, H, W // 2 + 1)
    return F.l1_loss(pred_fft.abs() * high_mask, target_fft.abs() * high_mask)


def total_variation_loss(foreground_prob: Tensor) -> Tensor:
    dx = (foreground_prob[:, :, :, 1:] - foreground_prob[:, :, :, :-1]).abs().mean()
    dy = (foreground_prob[:, :, 1:, :] - foreground_prob[:, :, :-1, :]).abs().mean()
    return dx + dy


def compute_loss(
    outputs: dict[str, Any],
    mask: Tensor,
    boundary_target: Tensor,
    cfg: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    seg_logits = outputs["seg_logits"]
    boundary_logits = outputs["boundary_logits"]
    num_classes = int(cfg["num_classes"])
    weights = cfg.get("loss_weights", {})

    ce = F.cross_entropy(seg_logits, mask.long())
    dice = foreground_dice_loss(seg_logits, mask, num_classes)
    boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target.float())
    boundary_dice = boundary_dice_loss(boundary_logits, boundary_target)
    bfreq = boundary_frequency_loss(boundary_logits, boundary_target, int(cfg["num_bands"]))
    probs = torch.softmax(seg_logits, dim=1)
    tv = total_variation_loss(probs[:, 1:].sum(dim=1, keepdim=True))
    gate_reg = outputs["gate_reg"]
    hf_ratio_penalty = outputs["hf_ratio_penalty"]

    loss = (
        ce
        + dice
        + float(weights.get("boundary_bce", 0.50)) * boundary_bce
        + float(weights.get("boundary_dice", 0.30)) * boundary_dice
        + float(weights.get("boundary_frequency", 0.20)) * bfreq
        + float(weights.get("tv", 0.05)) * tv
        + float(weights.get("gate_reg", 0.03)) * gate_reg
        + float(weights.get("hf_ratio", 0.005)) * hf_ratio_penalty
    )
    return loss, {
        "ce": float(ce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "boundary_bce": float(boundary_bce.detach().cpu()),
        "boundary_dice": float(boundary_dice.detach().cpu()),
        "boundary_frequency": float(bfreq.detach().cpu()),
        "tv": float(tv.detach().cpu()),
        "gate_reg": float(gate_reg.detach().cpu()),
        "hf_ratio_penalty": float(hf_ratio_penalty.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    split: str,
    log_ssr: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    num_classes = int(cfg["num_classes"])
    totals: dict[str, float] = {}
    total_samples = 0
    inter = torch.zeros(num_classes, device=device)
    denom = torch.zeros(num_classes, device=device)
    ssr_rows: list[dict[str, Any]] = []
    detailed_logs: dict[str, Any] | None = None

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            boundary_target = boundary_map_from_mask(masks).to(device)
            return_logs = log_ssr and batch_idx == 0

            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images, boundary_mask=boundary_target, return_logs=return_logs)
            loss, parts = compute_loss(outputs, masks, boundary_target, cfg)
            if training:
                loss.backward()
                grad_clip = float(cfg.get("grad_clip", 0.0) or 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_size = int(images.shape[0])
            total_samples += batch_size
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size

            preds = outputs["seg_logits"].argmax(dim=1)
            for cls in range(num_classes):
                pred_c = preds == cls
                target_c = masks == cls
                inter[cls] += (pred_c & target_c).sum()
                denom[cls] += pred_c.sum() + target_c.sum()

            if return_logs:
                detailed_logs = outputs.get("logs")
                ssr_rows.extend(flatten_ssr_logs(epoch, split, detailed_logs))

    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    metrics = {key: value / max(total_samples, 1) for key, value in totals.items()}
    for cls, name in enumerate(CLASS_NAMES[:num_classes]):
        metrics[f"dice_{name}"] = float(dice[cls].detach().cpu())
    fg_keys = [f"dice_{name}" for name in CLASS_NAMES[1:num_classes]]
    metrics["fg_dice"] = float(np.mean([metrics[k] for k in fg_keys])) if fg_keys else metrics["dice_BG"]
    return metrics, ssr_rows, detailed_logs


def flatten_ssr_logs(epoch: int, split: str, logs: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not logs:
        return rows
    for block, block_logs in logs.items():
        for metric, value in block_logs.items():
            if isinstance(value, list):
                for band, band_value in enumerate(value):
                    rows.append({"epoch": epoch, "split": split, "block": block, "metric": metric, "band": band, "value": float(band_value)})
            else:
                rows.append({"epoch": epoch, "split": split, "block": block, "metric": metric, "band": "", "value": float(value)})
    return rows


def print_detailed_logs(logs: dict[str, Any] | None, prefix: str) -> None:
    if not logs:
        return
    metrics = [
        "retain_gate_mean",
        "suppress_gate_mean",
        "update_gate_mean",
        "input_energy",
        "output_energy",
        "phase_coherence",
        "retain_contribution",
        "update_contribution",
        "suppress_contribution",
        "high_freq_ratio",
        "high_freq_penalty",
        "boundary_to_nonboundary_high_ratio",
        "gamma",
        "residual_gate_mean",
        "residual_gate_std",
    ]
    for block, block_logs in logs.items():
        print(f"  {prefix}/{block}")
        for metric in metrics:
            if metric in block_logs:
                print(f"    {metric}: {block_logs[metric]}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(run_dir: Path, rows: list[dict[str, Any]], ssr_rows: list[dict[str, Any]]) -> None:
    prepare_plot_cache(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    if rows:
        epochs = [row["epoch"] for row in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row["train_loss"] for row in rows], label="train")
        plt.plot(epochs, [row["val_loss"] for row in rows], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row["train_fg_mean"] for row in rows], label="train fg mean")
        plt.plot(epochs, [row["val_fg_mean"] for row in rows], label="val fg mean")
        plt.xlabel("epoch")
        plt.ylabel("foreground Dice")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "val_dice_curve.png", dpi=160)
        plt.close()

    _plot_ssr_metrics(run_dir, ssr_rows)


def _plot_ssr_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    def plot_metric_set(metrics: list[str], filename: str, title: str) -> None:
        selected = [row for row in rows if row["split"] == "val" and row["metric"] in metrics]
        if not selected:
            return
        plt.figure(figsize=(9, 5))
        for metric in metrics:
            for block in sorted({row["block"] for row in selected}):
                bands = sorted({row["band"] for row in selected if row["metric"] == metric and row["block"] == block}, key=lambda x: -1 if x == "" else int(x))
                for band in bands:
                    series = [row for row in selected if row["metric"] == metric and row["block"] == block and row["band"] == band]
                    series.sort(key=lambda row: row["epoch"])
                    label = f"{block}:{metric}:b{band}" if band != "" else f"{block}:{metric}"
                    plt.plot([row["epoch"] for row in series], [row["value"] for row in series], marker="o", linewidth=1.4, label=label)
        plt.title(title)
        plt.xlabel("epoch")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=160)
        plt.close()

    plot_metric_set(["retain_gate_mean", "update_gate_mean", "suppress_gate_mean"], "gate_curves.png", "SSR gate means")
    plot_metric_set(["retain_contribution", "update_contribution", "suppress_contribution"], "contribution_curves.png", "SSR contributions")
    plot_metric_set(["high_freq_ratio"], "high_freq_ratio_curves.png", "High-frequency ratio")
    plot_metric_set(["high_freq_penalty"], "high_freq_penalty_curves.png", "High-frequency penalty")
    plot_metric_set(["boundary_to_nonboundary_high_ratio"], "boundary_ratio_curves.png", "Boundary/non-boundary high ratio")
    plot_metric_set(["gamma"], "gamma_curves.png", "Effective gamma")
    plot_metric_set(["residual_gate_mean"], "residual_gate_mean_curves.png", "Residual gate mean")


def save_prediction_grid(model: nn.Module, loader: DataLoader, device: torch.device, path: Path) -> None:
    prepare_plot_cache(path.parent)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    boundary_target = boundary_map_from_mask(masks).to(device)
    with torch.no_grad():
        outputs = model(images, boundary_mask=boundary_target)
        preds = outputs["seg_logits"].argmax(dim=1)
        boundary_prob = torch.sigmoid(outputs["boundary_logits"])
        errors = (preds != masks).float()

    n = min(4, images.shape[0])
    fig, axes = plt.subplots(n, 5, figsize=(13, 2.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n):
        img = images[i, images.shape[1] // 2].detach().cpu().numpy()
        panels = [
            (img, "image", "gray", None, None),
            (masks[i].detach().cpu().numpy(), "mask", "viridis", 0, 3),
            (preds[i].detach().cpu().numpy(), "prediction", "viridis", 0, 3),
            (boundary_prob[i, 0].detach().cpu().numpy(), "boundary prob", "magma", 0, 1),
            (errors[i].detach().cpu().numpy(), "error", "Reds", 0, 1),
        ]
        for ax, (data, title, cmap, vmin, vmax) in zip(axes[i], panels):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def prepare_plot_cache(run_dir: Path) -> None:
    cache_root = run_dir / ".plot_cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


def is_cuda_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def train_once(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))
    if device.type == "cpu" and int(cfg.get("num_workers", 0)) > 0:
        print("CPU run detected; using num_workers=0 to avoid local DataLoader worker shared-memory failures.")
        cfg["num_workers"] = 0

    run_dir = Path(cfg["output_root"]) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["actual_batch_size"] = int(cfg["batch_size"])

    train_loader, val_loader, split_info = make_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    model_params = count_parameters(model)

    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    training_rows: list[dict[str, Any]] = []
    ssr_rows: list[dict[str, Any]] = []
    best_val = -1.0
    best_epoch = 0
    best_row: dict[str, Any] | None = None

    print(
        f"SSR phase2 run={cfg['run_name']} variant={cfg.get('variant')} "
        f"device={device} batch={cfg['actual_batch_size']} image={cfg['image_size']} "
        f"train_slices={split_info['train_slices']} val_slices={split_info['val_slices']}"
    )

    for epoch in range(1, int(cfg["epochs"]) + 1):
        log_ssr = epoch == 1 or epoch % 5 == 0 or epoch == int(cfg["epochs"])
        train_metrics, train_ssr_rows, train_logs = run_epoch(model, train_loader, optimizer, device, cfg, epoch, "train", log_ssr)
        val_metrics, val_ssr_rows, val_logs = run_epoch(model, val_loader, None, device, cfg, epoch, "val", log_ssr)
        ssr_rows.extend(train_ssr_rows)
        ssr_rows.extend(val_ssr_rows)

        row = {
            "epoch": epoch,
            "actual_batch_size": int(cfg["actual_batch_size"]),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_fg_mean": train_metrics["fg_dice"],
            "val_fg_mean": val_metrics["fg_dice"],
            "train_fg_dice": train_metrics["fg_dice"],
            "val_fg_dice": val_metrics["fg_dice"],
            "val_dice_BG": val_metrics.get("dice_BG", 0.0),
            "val_dice_RV": val_metrics.get("dice_RV", 0.0),
            "val_dice_MYO": val_metrics.get("dice_MYO", 0.0),
            "val_dice_LV": val_metrics.get("dice_LV", 0.0),
            "train_boundary_bce": train_metrics["boundary_bce"],
            "train_boundary_dice": train_metrics["boundary_dice"],
            "train_bfreq": train_metrics["boundary_frequency"],
            "train_tv": train_metrics["tv"],
            "train_gate_reg": train_metrics["gate_reg"],
            "train_hf_ratio_penalty": train_metrics["hf_ratio_penalty"],
            "val_boundary_bce": val_metrics["boundary_bce"],
            "val_boundary_dice": val_metrics["boundary_dice"],
            "val_bfreq": val_metrics["boundary_frequency"],
            "val_tv": val_metrics["tv"],
            "val_gate_reg": val_metrics["gate_reg"],
            "val_hf_ratio_penalty": val_metrics["hf_ratio_penalty"],
        }
        training_rows.append(row)

        if val_metrics["fg_dice"] > best_val:
            best_val = val_metrics["fg_dice"]
            best_epoch = epoch
            best_row = dict(row)
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_fg_mean": best_val, "config": cfg}, run_dir / "best_model.pt")

        write_csv(run_dir / "training_log.csv", training_rows)
        write_csv(run_dir / "ssr_logs.csv", ssr_rows, ["epoch", "split", "block", "metric", "band", "value"])

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_fg={train_metrics['fg_dice']:.4f} val_fg={val_metrics['fg_dice']:.4f} "
            f"RV={val_metrics.get('dice_RV', 0.0):.4f} MYO={val_metrics.get('dice_MYO', 0.0):.4f} LV={val_metrics.get('dice_LV', 0.0):.4f} "
            f"bfreq={val_metrics['boundary_frequency']:.4f} hf_pen={val_metrics['hf_ratio_penalty']:.4f} "
            f"gate_reg={val_metrics['gate_reg']:.5f}"
        )
        if log_ssr:
            print_detailed_logs(train_logs, "train")
            print_detailed_logs(val_logs, "val")

    torch.save({"epoch": int(cfg["epochs"]), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg}, run_dir / "final_model.pt")
    plot_training_curves(run_dir, training_rows, ssr_rows)
    save_prediction_grid(model, train_loader, device, run_dir / "train_predictions.png")
    save_prediction_grid(model, val_loader, device, run_dir / "val_predictions.png")

    final_row = training_rows[-1]
    summary = {
        "run_name": cfg["run_name"],
        "variant": cfg.get("variant", "default"),
        "best_epoch": best_epoch,
        "best_val_fg_mean": best_val,
        "best_val_fg_dice": best_val,
        "best_val_dice_RV": (best_row or {}).get("val_dice_RV", 0.0),
        "best_val_dice_MYO": (best_row or {}).get("val_dice_MYO", 0.0),
        "best_val_dice_LV": (best_row or {}).get("val_dice_LV", 0.0),
        "final_val_fg_mean": final_row["val_fg_mean"],
        "actual_batch_size": int(cfg["actual_batch_size"]),
        "image_size": int(cfg["image_size"]),
        "epochs": int(cfg["epochs"]),
        "model_parameter_count": model_params,
        "train_slices": split_info["train_slices"],
        "val_slices": split_info["val_slices"],
        "train_cases": len(split_info["train_cases"]),
        "val_cases": len(split_info["val_cases"]),
        "config_flags": {
            "residual_gate_type": cfg["ssr"].get("residual_gate_type"),
            "geometry_refine": cfg["ssr"].get("geometry_refine"),
            "use_bounded_gamma": cfg["ssr"].get("use_bounded_gamma"),
            "gamma_max": cfg["ssr"].get("gamma_max"),
            "suppress_min": cfg["ssr"].get("suppress_min"),
            "suppress_max": cfg["ssr"].get("suppress_max"),
            "hf_ratio_threshold": cfg["ssr"].get("hf_ratio_threshold"),
            "loss_weights": cfg.get("loss_weights", {}),
        },
        "artifacts": [
            "training_log.csv",
            "ssr_logs.csv",
            "loss_curve.png",
            "val_dice_curve.png",
            "gate_curves.png",
            "contribution_curves.png",
            "high_freq_ratio_curves.png",
            "high_freq_penalty_curves.png",
            "boundary_ratio_curves.png",
            "gamma_curves.png",
            "train_predictions.png",
            "val_predictions.png",
            "best_model.pt",
            "final_model.pt",
            "config_resolved.yaml",
        ],
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Finished SSR phase2 run. Best val fg mean={best_val:.4f} at epoch {best_epoch}.")
    print(f"Artifacts saved under {run_dir}")
    return summary


def train_with_oom_recovery(cfg: dict[str, Any]) -> dict[str, Any]:
    batch_size = int(cfg["batch_size"])
    while batch_size >= 1:
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["batch_size"] = batch_size
        try:
            return train_once(trial_cfg)
        except RuntimeError as exc:
            if is_cuda_oom(exc) and torch.cuda.is_available():
                print(f"CUDA OOM at batch_size={batch_size}; retrying with batch_size={batch_size // 2}.")
                torch.cuda.empty_cache()
                batch_size //= 2
                if batch_size < 1:
                    raise RuntimeError("CUDA OOM even at batch_size=1.") from exc
                continue
            raise
    raise RuntimeError("CUDA OOM recovery exhausted all batch sizes.")


def main() -> None:
    args = parse_args()
    cfg = apply_variant_and_cli(load_config(args.config), args)
    train_with_oom_recovery(cfg)


if __name__ == "__main__":
    main()
