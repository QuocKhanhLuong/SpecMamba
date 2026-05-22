#!/usr/bin/env python3
"""Analyze SSRBlockV3 diagnostic logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SSR debug logs")
    parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("epoch", "band", "value"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def load_update_budget(run_dir: Path) -> float:
    cfg_path = run_dir / "config_resolved.yaml"
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("ssr", {}).get("update_budget", 1.5))
    summary = run_dir / "summary.json"
    if summary.exists():
        with open(summary, encoding="utf-8") as f:
            json.load(f)
    return 1.5


def plot_curves(run_dir: Path, ssr_rows: list[dict[str, Any]]) -> None:
    prepare_plot_cache(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    def plot(metrics: list[str], filename: str, title: str) -> None:
        rows = [r for r in ssr_rows if r["split"] == "val" and r["metric"] in metrics]
        if not rows:
            return
        plt.figure(figsize=(9, 5))
        for metric in metrics:
            blocks = sorted({r["block"] for r in rows if r["metric"] == metric})
            for block in blocks:
                bands = sorted({r["band"] for r in rows if r["metric"] == metric and r["block"] == block})
                for band in bands:
                    series = [
                        r for r in rows
                        if r["metric"] == metric and r["block"] == block and r["band"] == band
                    ]
                    series.sort(key=lambda r: r["epoch"])
                    label = f"{block}:{metric}:b{int(band)}" if band != "" else f"{block}:{metric}"
                    plt.plot(
                        [r["epoch"] for r in series],
                        [r["value"] for r in series],
                        marker="o",
                        linewidth=1.4,
                        label=label,
                    )
        plt.title(title)
        plt.xlabel("epoch")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=160)
        plt.close()

    plot(["retain_gate_mean", "update_gate_mean", "suppress_gate_mean"], "analysis_gate_curves.png", "Gate curves")
    plot(["update_contribution"], "analysis_update_contribution_curves.png", "Update contribution")
    plot(["suppress_contribution"], "analysis_suppress_contribution_curves.png", "Suppress contribution")
    plot(["boundary_to_nonboundary_high_ratio"], "analysis_boundary_high_ratio_curves.png", "Boundary high ratio")
    plot(["high_freq_ratio"], "analysis_high_freq_ratio_curves.png", "High-frequency ratio")


def prepare_plot_cache(run_dir: Path) -> None:
    cache_root = run_dir / ".plot_cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache.resolve()))


def values(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    band: int | None = None,
    split: str = "val",
) -> list[float]:
    out = []
    for row in rows:
        if row["split"] != split or row["metric"] != metric:
            continue
        if band is not None and int(row["band"]) != band:
            continue
        out.append(float(row["value"]))
    return out


def improving(values_: list[float]) -> bool:
    if len(values_) < 2:
        return False
    return values_[-1] > values_[0] * 1.05


def print_diagnosis(training_rows: list[dict[str, Any]], ssr_rows: list[dict[str, Any]], update_budget: float) -> None:
    print("SSR diagnosis")
    if training_rows:
        last = training_rows[-1]
        print(
            f"- Final val foreground Dice: {float(last.get('val_fg_dice', 0.0)):.4f}; "
            f"val loss: {float(last.get('val_loss', 0.0)):.4f}"
        )

    update_b0 = values(ssr_rows, "update_gate_mean", band=0)
    collapse_threshold = 0.75 * update_budget
    collapse_count = sum(v > collapse_threshold for v in update_b0)
    if update_b0:
        print(
            "- Update gate collapse to band 0: "
            f"{'YES' if collapse_count > len(update_b0) / 2 else 'no'} "
            f"({collapse_count}/{len(update_b0)} snapshots above {collapse_threshold:.3f})"
        )

    suppress_hi = values(ssr_rows, "suppress_gate_mean", band=2) + values(ssr_rows, "suppress_gate_mean", band=3)
    saturation_count = sum(v > 0.48 for v in suppress_hi)
    if suppress_hi:
        print(
            "- Suppress gate saturation in bands 2/3: "
            f"{'YES' if saturation_count > len(suppress_hi) / 2 else 'no'} "
            f"({saturation_count}/{len(suppress_hi)} snapshots above 0.48)"
        )

    high_ratios = values(ssr_rows, "high_freq_ratio")
    if high_ratios:
        low_count = sum(v < 0.75 for v in high_ratios)
        high_count = sum(v > 1.7 for v in high_ratios)
        print(
            "- High-frequency ratio: "
            f"collapse={'YES' if low_count > len(high_ratios) / 2 else 'no'} "
            f"({low_count}/{len(high_ratios)} < 0.75), "
            f"amplification_risk={'YES' if high_count > len(high_ratios) / 2 else 'no'} "
            f"({high_count}/{len(high_ratios)} > 1.7)"
        )

    boundary_ratios = values(ssr_rows, "boundary_to_nonboundary_high_ratio")
    if boundary_ratios:
        print(
            "- Boundary/non-boundary high-ratio trend: "
            f"{'improving' if improving(boundary_ratios) else 'not clearly improving'} "
            f"(first={boundary_ratios[0]:.4f}, last={boundary_ratios[-1]:.4f})"
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    training_rows = read_csv(run_dir / "training_log.csv")
    ssr_rows = read_csv(run_dir / "ssr_logs.csv")
    update_budget = load_update_budget(run_dir)
    plot_curves(run_dir, ssr_rows)
    print_diagnosis(training_rows, ssr_rows, update_budget)
    print(f"Analysis plots saved under {run_dir}")


if __name__ == "__main__":
    main()
