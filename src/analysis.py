"""Batch-comparable EDA for the SKALA ESS Health mini-project."""

from __future__ import annotations

import argparse
import gc
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = {
    "Batch 1": ROOT / "data/raw/2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": ROOT / "data/raw/2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": ROOT / "data/raw/2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
PALETTE = {"Batch 1": "#2864DC", "Batch 2": "#1B9A8A", "Batch 3": "#7554B8"}
LIFE_COLORS = {"Short (<500)": "#E07A5F", "Middle (500-1000)": "#E9B949", "Long (>1000)": "#2A9D8F", "No EOL label": "#8B95A5"}
MAX_PLAUSIBLE_QD_AH = 1.3


def as_list(value):
    """Normalize MATLAB singleton arrays and mat73 dict-of-arrays values."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return list(value.reshape(-1))
        if value.ndim == 0:
            return [value.item()]
        return list(value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def scalar(value, default=np.nan):
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return item.item() if hasattr(item, "item") else item


def text_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "US":
            return "".join(map(str, value.reshape(-1))).strip()
        if value.dtype == object and value.size == 1:
            return text_value(value.reshape(-1)[0])
    return str(value).strip()


def to_cell_list(batch):
    if isinstance(batch, dict):
        # mat73 can represent a MATLAB struct array as a dict of parallel arrays.
        lengths = []
        for value in batch.values():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                if isinstance(value, dict):
                    continue
                lengths.append(len(as_list(value)))
        if not lengths:
            raise ValueError("MATLAB batch has no recognizable cell records")
        n_cells = max(lengths)
        cells = []
        for i in range(n_cells):
            cell = {}
            for key, value in batch.items():
                if isinstance(value, dict):
                    cell[key] = {k: (as_list(v)[i] if len(as_list(v)) == n_cells else v) for k, v in value.items()}
                else:
                    vals = as_list(value)
                    cell[key] = vals[i] if len(vals) == n_cells else value
            cells.append(cell)
        return cells
    if isinstance(batch, np.ndarray):
        return list(batch.reshape(-1))
    return list(batch)


def cycles_as_list(cycles):
    if isinstance(cycles, dict):
        fields = {key: as_list(value) for key, value in cycles.items()}
        n = max((len(v) for v in fields.values()), default=0)
        return [{key: values[i] for key, values in fields.items() if i < len(values)} for i in range(n)]
    if isinstance(cycles, np.ndarray):
        return list(cycles.reshape(-1))
    return list(cycles) if isinstance(cycles, (list, tuple)) else []


def vector(summary, key):
    if isinstance(summary, dict) and key in summary:
        try:
            return np.asarray(summary[key], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            pass
    return np.array([], dtype=float)


def parse_policy(policy):
    match = re.search(r"([\d.]+)\s*C\s*\(\s*(\d+)\s*%\s*\)\s*-\s*([\d.]+)\s*C", policy, re.I)
    if not match:
        match = re.search(r"([\d.]+)\s*C\s*\(\s*(\d+)\s*%\s*\).*?([\d.]+)\s*C", policy, re.I)
    if not match:
        return np.nan, np.nan, np.nan
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def delta_q_features(cycles):
    cycles = cycles_as_list(cycles)
    if len(cycles) < 100:
        return None, None
    q10 = np.asarray(cycles[9].get("Qdlin", []), dtype=float).reshape(-1)
    q100 = np.asarray(cycles[99].get("Qdlin", []), dtype=float).reshape(-1)
    n = min(q10.size, q100.size)
    if n < 100:
        return None, None
    delta = q100[:n] - q10[:n]
    voltage = np.linspace(2.0, 3.6, n)
    abs_delta = np.abs(delta)
    peak = int(np.nanargmax(abs_delta)) if np.isfinite(abs_delta).any() else 0
    feats = {
        "dq_mean": float(np.nanmean(delta)),
        "dq_std": float(np.nanstd(delta)),
        "dq_mean_abs": float(np.nanmean(abs_delta)),
        "dq_max_abs": float(np.nanmax(abs_delta)),
        "dq_area_abs": float(np.trapezoid(abs_delta, voltage)),
        "dq_peak_voltage": float(voltage[peak]),
        "dq_low_v_mean_abs": float(np.nanmean(abs_delta[voltage < 2.8])),
        "dq_high_v_mean_abs": float(np.nanmean(abs_delta[voltage >= 2.8])),
    }
    return feats, (voltage, delta)


def load_batch(path, batch_name):
    try:
        import mat73
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before loading the MATLAB v7.3 batch files") from exc
    logging.disable(logging.CRITICAL)
    mat = mat73.loadmat(str(path))
    if "batch" not in mat:
        raise ValueError(f"No 'batch' variable found in {path.name}; keys={list(mat)}")
    cells = to_cell_list(mat["batch"])
    summaries, cycles_out, dq_curves = [], [], []
    for idx, cell in enumerate(cells):
        summary = cell.get("summary", {})
        life_value = scalar(cell.get("cycle_life", np.nan))
        life_number = pd.to_numeric(pd.Series([life_value]), errors="coerce").iloc[0]
        life = int(life_number) if np.isfinite(life_number) else np.nan
        policy = text_value(cell.get("policy_readable", cell.get("policy", "unknown")))
        c1, soc_switch, c2 = parse_policy(policy)
        vectors = {
            "cycle": vector(summary, "cycle"),
            "QD": vector(summary, "QDischarge"),
            "QC": vector(summary, "QCharge"),
            "IR": vector(summary, "IR"),
            "Tmax": vector(summary, "Tmax"),
            "Tavg": vector(summary, "Tavg"),
            "Tmin": vector(summary, "Tmin"),
            "chargetime": vector(summary, "chargetime"),
            "discharge_time": vector(summary, "discharge_time"),
        }
        sizes = [len(v) for v in vectors.values() if len(v)]
        n_cycles = min(sizes) if sizes else 0
        row = {
            "batch": batch_name,
            "cell_id": f"{batch_name.replace(' ', '')}_{idx + 1:03d}",
            "cell_index": idx,
            "cycle_life": life,
            "charging_policy": policy,
            "C1": c1,
            "SOC_switch": soc_switch,
            "C2": c2,
            "n_cycles": n_cycles,
        }
        for key, arr in vectors.items():
            start = 1 if key == "QD" and len(arr) > 1 and np.isfinite(arr[0]) and arr[0] <= 0 else 0
            use = arr[start: min(100, len(arr))]
            valid = np.isfinite(use)
            finite = use[valid]
            row[f"early_{key}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            row[f"early_{key}_std"] = float(np.std(finite)) if finite.size else np.nan
            if finite.size >= 2:
                cycles_idx = np.arange(start + 1, start + 1 + len(use))[valid]
                row[f"early_{key}_slope"] = float(np.polyfit(cycles_idx, finite, 1)[0])
            else:
                row[f"early_{key}_slope"] = np.nan
        for k in range(n_cycles):
            cycles_out.append({
                "batch": batch_name,
                "cell_id": row["cell_id"],
                "cycle": k + 1,
                **{name: float(arr[k]) for name, arr in vectors.items() if k < len(arr) and np.isfinite(arr[k])},
            })
        feats, curve = delta_q_features(cell.get("cycles", []))
        if feats:
            row.update(feats)
            voltage, delta = curve
            dq_curves.append(pd.DataFrame({"batch": batch_name, "cell_id": row["cell_id"], "cycle_life": life, "voltage": voltage, "delta_q": delta}))
        summaries.append(row)
    del mat, cells
    gc.collect()
    return pd.DataFrame(summaries), pd.DataFrame(cycles_out), pd.concat(dq_curves, ignore_index=True) if dq_curves else pd.DataFrame()


def describe_batches(cells):
    return (cells.groupby("batch")["cycle_life"]
            .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
            .round(1))


def life_group(values):
    values = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select([values.isna(), values < 500, values > 1000], ["No EOL label", "Short (<500)", "Long (>1000)"], default="Middle (500-1000)"), index=values.index)


def plot_life_distribution(cells, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.1))
    sns.histplot(data=cells, x="cycle_life", hue="batch", bins=18, element="step", stat="count", common_norm=False, palette=PALETTE, ax=axes[0])
    axes[0].axvline(500, color="#E07A5F", ls="--", lw=1, label="500 cycles")
    axes[0].axvline(1000, color="#2A9D8F", ls="--", lw=1, label="1,000 cycles")
    axes[0].set(title="Cycle life distribution by batch", xlabel="Cycle life (cycles)", ylabel="Cells")
    axes[0].set_xlim(150, 2300)
    axes[0].legend(frameon=False, fontsize=8)
    sns.boxplot(data=cells, x="batch", y="cycle_life", hue="batch", palette=PALETTE, legend=False, ax=axes[1])
    axes[1].set(title="Spread and outliers", xlabel="", ylabel="Cycle life (cycles)")
    fig.tight_layout()
    fig.savefig(out / "01_cycle_life_by_batch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_degradation(cycles, cells, out):
    merged = cycles.merge(cells[["cell_id", "batch", "cycle_life"]], on=["cell_id", "batch"], suffixes=("", "_cell"))
    # Cycle 1 is encoded as a zero-capacity initialization in these files.
    merged = merged[(merged["cycle"].between(2, 500)) & merged["QD"].between(0, MAX_PLAUSIBLE_QD_AH)]
    merged["life_group"] = life_group(merged["cycle_life"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 7.0), sharey=True)
    for ax, batch in zip(axes, PALETTE):
        part = merged[merged.batch == batch]
        for group, subset in part.groupby("life_group", observed=True):
            stats = subset.groupby("cycle").QD.agg(
                median="median", low=lambda x: x.quantile(.1), high=lambda x: x.quantile(.9)
            )
            color = LIFE_COLORS[group]
            ax.plot(stats.index, stats["median"], color=color, label=group, lw=1.5)
            ax.fill_between(stats.index, stats["low"], stats["high"], color=color, alpha=.13, linewidth=0)
        ax.set(title=batch, xlabel="Cycle", ylabel="Discharge capacity Qd (Ah)")
        ax.legend(frameon=False, fontsize=7, title="Cycle-life group")
    fig.suptitle("Discharge-capacity degradation (median and 80% interval)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "02_qd_degradation_by_batch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_q(curves, out):
    if curves.empty:
        return
    curves = curves.copy()
    curves["life_group"] = life_group(curves["cycle_life"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 7.8), sharey=True)
    for ax, batch in zip(axes, PALETTE):
        part = curves[curves.batch == batch]
        for group, subset in part.groupby("life_group", observed=True):
            stats = subset.groupby("voltage").delta_q.agg(
                median="median", low=lambda x: x.quantile(.1), high=lambda x: x.quantile(.9)
            )
            color = LIFE_COLORS[group]
            ax.plot(stats.index, stats["median"], color=color, label=group, lw=1.5)
            ax.fill_between(stats.index, stats["low"], stats["high"], color=color, alpha=.13, linewidth=0)
        ax.axhline(0, color="#667085", lw=.8)
        ax.set(title=batch, xlabel="Voltage (V)", ylabel="ΔQ(V): Qd(100) - Qd(10) (Ah)")
        ax.legend(frameon=False, fontsize=7, title="Cycle-life group")
    fig.suptitle("Early-cycle voltage-capacity change by life group", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "03_delta_q_voltage_by_batch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_charging_policy(cells, out):
    policy = (cells.dropna(subset=["C1", "C2"]).groupby(["batch", "charging_policy"], as_index=False)
              .agg(mean_life=("cycle_life", "mean"), sd_life=("cycle_life", "std"), n=("cycle_life", "count"), C1=("C1", "first"), C2=("C2", "first"), SOC_switch=("SOC_switch", "first")))
    policy.to_csv(out / "charging_policy_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 7.0), sharey=True)
    for ax, batch in zip(axes, PALETTE):
        part = policy[policy.batch == batch].sort_values("mean_life")
        sns.barplot(data=part, x="charging_policy", y="mean_life", color=PALETTE[batch], ax=ax)
        ax.errorbar(np.arange(len(part)), part.mean_life, yerr=part.sd_life.fillna(0), fmt="none", ecolor="#334155", capsize=2, lw=.8)
        ax.set(title=batch, xlabel="Charging policy", ylabel="Mean cycle life")
        ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    fig.suptitle("Cycle life across charging policies (mean ± 1 SD)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "04_policy_vs_life.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(cells, out):
    candidates = [c for c in cells.columns if (c.startswith("early_") and not c.startswith("early_cycle_")) or c.startswith("dq_")]
    corr_rows = []
    for batch, part in cells.groupby("batch"):
        numeric = part[["cycle_life", *candidates]].apply(pd.to_numeric, errors="coerce")
        for feature in candidates:
            valid = numeric[["cycle_life", feature]].dropna()
            if len(valid) >= 4:
                corr_rows.append({"batch": batch, "feature": feature, "n": len(valid), "spearman_r": valid.corr(method="spearman").iloc[0, 1], "pearson_r": valid.corr(method="pearson").iloc[0, 1]})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out / "feature_cycle_life_correlations.csv", index=False)
    if corr.empty:
        return
    top_features = (corr.assign(abs_r=corr.spearman_r.abs()).groupby("feature").abs_r.max().nlargest(12).index)
    plot_data = corr[corr.feature.isin(top_features)].pivot(index="feature", columns="batch", values="spearman_r")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})
    sns.heatmap(plot_data, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=.5, ax=axes[0], cbar_kws={"label": "Spearman ρ"})
    axes[0].set(title="Early features vs cycle life", xlabel="Batch", ylabel="Feature")
    all_features = [c for c in candidates if c in cells and cells[c].notna().sum() >= 4]
    matrix = cells[all_features].corr(method="spearman")
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axes[1], cbar=False)
    axes[1].set(title="Candidate-feature redundancy", xlabel="Feature", ylabel="Feature")
    axes[1].tick_params(axis="x", labelrotation=75, labelsize=6)
    axes[1].tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "05_early_feature_relationships.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/eda")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells_all, cycles_all, dq_all = [], [], []
    for batch_name, default_path in DEFAULT_FILES.items():
        path = args.data_dir / default_path.name
        if not path.is_file():
            raise FileNotFoundError(f"Required assignment data file not found: {path}")
        print(f"Loading {batch_name}: {path.name}", flush=True)
        cells, cycles, dq = load_batch(path, batch_name)
        cells_all.append(cells)
        cycles_all.append(cycles)
        if not dq.empty:
            dq_all.append(dq)
        print(f"  {len(cells)} cells, {len(cycles):,} cycle records", flush=True)
    cells = pd.concat(cells_all, ignore_index=True)
    cycles = pd.concat(cycles_all, ignore_index=True)
    dq_curves = pd.concat(dq_all, ignore_index=True) if dq_all else pd.DataFrame()
    cells.to_csv(args.out_dir / "cell_features.csv", index=False)
    cycles.to_csv(args.out_dir / "cycle_summary.csv", index=False)
    if not dq_curves.empty:
        dq_curves.to_csv(args.out_dir / "delta_q_curves.csv", index=False)
    describe_batches(cells).to_csv(args.out_dir / "cycle_life_summary.csv")
    plot_life_distribution(cells, args.out_dir)
    plot_degradation(cycles, cells, args.out_dir)
    plot_delta_q(dq_curves, args.out_dir)
    plot_charging_policy(cells, args.out_dir)
    plot_correlations(cells, args.out_dir)
    print("\nCycle-life summary by batch:\n", describe_batches(cells).to_string())
    print(f"\nEDA artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
