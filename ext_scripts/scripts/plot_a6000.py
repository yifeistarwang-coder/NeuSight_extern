"""Render the A6000 supplement from audited summaries, without using a GPU."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from a6000_common import MATRIX, SUMMARY, read_json, sha256, workloads, write_json


MODES = ("inf", "train")
MODEL_LABELS = ("BERT-Large", "GPT-2 Large", "GPT-3 XL", "OPT-1.3B", "GPT-3 2.7B", "Switch")
CATEGORIES = ("BMM", "Linear", "VEC", "VECsoftmax", "VECln")
CATEGORY_LABELS = ("BMM", "FC", "EW", "Softmax", "LN")
BLUE, RED, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#9A9A9A"
DEVICE_LABELS = {
    "Tesla_P4": "P4\n8 GB",
    "Tesla_P100-PCIE-16GB": "P100\n16 GB",
    "Tesla_V100-PCIE-32GB": "V100\n32 GB",
    "Tesla_T4": "T4\n16 GB",
    "NVIDIA_A100-PCIE-40GB": "A100 PCIe\n40 GB",
    "NVIDIA_A100_80GB_PCIe": "A100 PCIe\n80 GB",
    "NVIDIA_L4": "L4\n24 GB",
    "NVIDIA_H100_80GB_HBM3": "H100\n80 GB",
    "NVIDIA_RTX_A6000": "RTX A6000\n48 GB",
}
SOURCE_FILES = (
    "e2e.csv", "metrics.csv", "operator_medians.csv", "operator_metrics.csv",
    "common_config_details.csv", "common_config_comparison.csv",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_close(actual, expected, context):
    require(np.allclose(actual, expected, rtol=1e-9, atol=1e-8, equal_nan=True),
            f"Inconsistent {context}: {actual} != {expected}")


def distribution(values):
    values = pd.Series(values, dtype=float)
    return {"valid_count": len(values), "mape_pct": values.mean(),
            "median_ape_pct": values.median(), "p90_ape_pct": values.quantile(.9)}


def validate_e2e(e2e):
    require(e2e["name"].is_unique, "Duplicate end-to-end configurations")
    expected = {item["name"] for item in workloads()}
    require(set(e2e["name"]) == expected, "End-to-end table does not match the 24-configuration matrix")
    for row in e2e.to_dict("records"):
        require(row["name"] == f"{row['model']}-{row['mode']}-{row['seqlen']}-{row['batch']}",
                "Configuration metadata does not match its name")
        require(row["prediction_status"] == "success" and
                np.isfinite(row["predicted_e2e_ms"]) and row["predicted_e2e_ms"] > 0,
                f"Missing or invalid prediction: {row['name']}")
        valid_runs = sum(row[f"run_{run:02d}_status"] == "success" for run in (1, 2, 3))
        require(valid_runs == row["valid_repeats"], f"Repeat count mismatch: {row['name']}")
        if row["status"] != "success":
            require(valid_runs < 3 and pd.isna(row["median_e2e_ape_pct"]) and
                    pd.isna(row["median_e2e_ms"]),
                    f"Failed/incomplete configuration must not have a primary APE: {row['name']}")
            continue
        require(valid_runs == 3, f"Incomplete successful configuration: {row['name']}")
        measured = np.array([row[f"run_{run:02d}_e2e_ms"] for run in (1, 2, 3)])
        require(np.isfinite(measured).all() and (measured > 0).all(),
                f"Non-positive or invalid measurement: {row['name']}")
        median = np.median(measured)
        errors = np.abs(row["predicted_e2e_ms"] - measured) / measured * 100
        for run, error in enumerate(errors, 1):
            check_close(row[f"run_{run:02d}_e2e_ape_pct"], error, f"repeat APE for {row['name']}")
        for statistic, value in (("median", median), ("min", measured.min()), ("max", measured.max())):
            check_close(row[f"{statistic}_e2e_ms"], value, f"{statistic} latency for {row['name']}")
        check_close(row["median_e2e_ape_pct"], abs(row["predicted_e2e_ms"] - median) / median * 100,
                    f"primary APE for {row['name']}")
    return pd.DataFrame([
        {"mode": mode, **distribution(e2e.loc[(e2e["mode"] == mode) &
         (e2e["status"] == "success"), "median_e2e_ape_pct"])} for mode in MODES
    ])


def operator_statistics(medians, e2e):
    selected = medians[medians["phase"].eq("e2e") & medians["category"].isin(CATEGORIES)].copy()
    successful = e2e[e2e["status"] == "success"].set_index("name")
    expected = {(name, category) for name in successful.index for category in CATEGORIES}
    require(not selected.duplicated(["name", "category"]).any() and
            set(zip(selected["name"], selected["category"])) == expected,
            "Operator categories do not match complete successful workloads")
    require(selected["mode"].eq(selected["name"].map(successful["mode"])).all(),
            "Operator modes do not match end-to-end workloads")
    require(selected["valid_repeats"].eq(3).all(), "Operator totals require three repeats")
    require(np.isfinite(selected[["predicted_ms", "median_measured_ms"]]).all().all() and
            selected[["predicted_ms", "median_measured_ms"]].ge(0).all().all(),
            "Invalid operator latency")
    valid = selected["median_measured_ms"] > 0
    errors = selected.loc[valid, "predicted_ms"] - selected.loc[valid, "median_measured_ms"]
    check_close(selected.loc[valid, "ape_pct"],
                errors.abs() / selected.loc[valid, "median_measured_ms"] * 100, "operator APE")
    require(selected.loc[~valid, "ape_pct"].isna().all(), "Zero operator labels must be excluded")
    selected["absolute_error_ms"] = (selected["predicted_ms"] - selected["median_measured_ms"]).abs()
    return pd.DataFrame([
        {"mode": mode, "category": category, **distribution(group["ape_pct"]),
         "mean_absolute_error_ms": group["absolute_error_ms"].mean()}
        for mode in MODES for category in CATEGORIES
        for group in [selected[valid & selected["mode"].eq(mode) & selected["category"].eq(category)]]
    ])


def common_statistics(details, e2e):
    details = details[details["scope"] == "all_gpu_common"].copy()
    require(not details.empty and not details.duplicated(["mode", "reference_device", "name"]).any(),
            "Missing or duplicate common-configuration data")
    successful = e2e[e2e["status"] == "success"].set_index("name")
    require(set(details["name"]) <= set(successful.index), "Comparison includes unsuccessful A6000 workloads")
    require(details["mode"].eq(details["name"].map(successful["mode"])).all(), "Comparison mode mismatch")
    require(np.isfinite(details[["a6000_ape_pct", "reference_ape_pct"]]).all().all() and
            details[["a6000_ape_pct", "reference_ape_pct"]].ge(0).all().all(), "Invalid comparison APE")
    check_close(details["a6000_ape_pct"], details["name"].map(successful["median_e2e_ape_pct"]),
                "A6000 common-configuration APE")
    rows = []
    for mode in MODES:
        mode_details = details[details["mode"] == mode]
        require(not mode_details.empty, f"No common configurations for {mode}")
        names = set(mode_details["name"])
        for device, group in mode_details.groupby("reference_device", sort=False):
            require(set(group["name"]) == names, f"Unequal workload sets for {mode}/{device}")
            require(device in DEVICE_LABELS, f"Unknown comparison GPU: {device}")
            rows.append({"mode": mode, "device": device, "valid_count": len(names),
                         "configurations": ";".join(sorted(names)), "mape_pct": group["reference_ape_pct"].mean()})
        rows.append({"mode": mode, "device": "NVIDIA_RTX_A6000", "valid_count": len(names),
                     "configurations": ";".join(sorted(names)),
                     "mape_pct": successful.loc[sorted(names), "median_e2e_ape_pct"].mean()})
    return pd.DataFrame(rows)


def load_data(directory):
    audit = read_json(directory / "audit.json")
    verification = read_json(directory / "verification.json")
    require(audit["all_formal_attempts_terminal"] and audit["frozen_inputs_unchanged"] and verification["passed"],
            "Generate and verify the complete A6000 summary before plotting")
    sources = {name: sha256(directory / name) for name in (*SOURCE_FILES, "audit.json", "verification.json")}
    for name in SOURCE_FILES:
        require(sources[name] == audit["output_tables"][name]["sha256"], f"Unaudited/modified source: {name}")
    tables = {name: pd.read_csv(directory / name) for name in SOURCE_FILES}
    e2e = tables["e2e.csv"]
    metrics = validate_e2e(e2e)
    require(int(e2e["valid_repeats"].sum()) == audit["successful_attempts"] and
            int(e2e["status"].eq("success").sum()) == audit["successful_configurations"], "Audit count mismatch")
    for row in metrics.to_dict("records"):
        recorded = tables["metrics.csv"]
        recorded = recorded[recorded["mode"].eq(row["mode"]) & recorded["phase"].eq("e2e")]
        require(len(recorded) == 1, "Missing or duplicate end-to-end metrics")
        for key in ("valid_count", "mape_pct", "median_ape_pct", "p90_ape_pct"):
            check_close(row[key], recorded.iloc[0][key], f"{row['mode']} {key}")
    operators = operator_statistics(tables["operator_medians.csv"], e2e)
    recorded_ops = tables["operator_metrics.csv"]
    recorded_ops = recorded_ops[recorded_ops["group_scope"].eq("all_models") & recorded_ops["phase"].eq("e2e")]
    for row in operators.to_dict("records"):
        recorded = recorded_ops[(recorded_ops["mode"] == row["mode"]) & (recorded_ops["category"] == row["category"])]
        require(len(recorded) == 1, "Missing or duplicate operator metrics")
        for key in ("valid_count", "mape_pct", "mean_absolute_error_ms"):
            check_close(row[key], recorded.iloc[0][key], f"{row['mode']}/{row['category']} {key}")
    common = common_statistics(tables["common_config_details.csv"], e2e)
    recorded_common = tables["common_config_comparison.csv"]
    recorded_common = recorded_common[recorded_common["scope"] == "all_gpu_common"]
    for row in recorded_common.to_dict("records"):
        reference = common[(common["mode"] == row["mode"]) & (common["device"] == row["reference_device"])]
        a6000 = common[(common["mode"] == row["mode"]) & common["device"].eq("NVIDIA_RTX_A6000")]
        require(len(reference) == 1 and row["configurations"] == reference.iloc[0]["configurations"],
                "Common-configuration list mismatch")
        check_close(row["valid_count"], reference.iloc[0]["valid_count"], "common sample count")
        check_close(row["reference_mape_pct"], reference.iloc[0]["mape_pct"], "reference common MAPE")
        check_close(row["a6000_mape_pct"], a6000.iloc[0]["mape_pct"], "A6000 common MAPE")
    return e2e, metrics, operators, common, sources


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.tick_params(length=3, color="#888888")


def plot_e2e(e2e, metrics):
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 7.8))
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.15, hspace=.53)
    fig.suptitle("RTX A6000: end-to-end prediction error", fontsize=16, y=.98)
    fig.legend(handles=[Patch(facecolor=BLUE, label="Lower batch"),
                        Patch(facecolor=RED, hatch="//", label="Higher batch"),
                        Line2D([], [], color="#333333", linestyle="--", label="Mode MAPE")],
               loc="upper center", bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
    indexed = e2e.set_index("name")
    repeat_errors = e2e[[f"run_{run:02d}_e2e_ape_pct" for run in (1, 2, 3)]].to_numpy()
    limit = max(30, 10 * np.ceil(np.nanmax(repeat_errors) * 1.2 / 10))
    for ax, mode, panel in zip(axes, MODES, ("a", "b")):
        stats = metrics[metrics["mode"] == mode].iloc[0]
        title = "Inference" if mode == "inf" else "Training (forward + backward)"
        ax.set_title(f"({panel}) {title}: {int(stats['valid_count'])}/12 valid | "
                     f"MAPE {stats['mape_pct']:.2f}% | P90 APE {stats['p90_ape_pct']:.2f}%",
                     loc="left", pad=12, fontsize=11)
        for position, (model, (seqlen, inference, training)) in enumerate(MATRIX.items()):
            batches = inference if mode == "inf" else training
            for rank, (batch, color) in enumerate(zip(batches, (BLUE, RED))):
                row = indexed.loc[f"{model}-{mode}-{seqlen}-{batch}"]
                x = position + (-.19 if rank == 0 else .19)
                if row["status"] != "success":
                    ax.plot(x, 1.1, marker="x", color=color, markersize=6)
                    ax.text(x, 3.0, str(row["status"]).upper(), ha="center", va="bottom", rotation=90,
                            color="#666666", fontsize=8)
                else:
                    value = row["median_e2e_ape_pct"]
                    errors = [row[f"run_{run:02d}_e2e_ape_pct"] for run in (1, 2, 3)]
                    ax.bar(x, value, width=.33, color=color, hatch="//" if rank else None,
                           edgecolor="white", linewidth=.6, zorder=3)
                    ax.errorbar(x, value, yerr=[[value - min(errors)], [max(errors) - value]],
                                fmt="none", ecolor="#333333", capsize=2.5, linewidth=.8, zorder=4)
                    ax.text(x, max(errors) + .7, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
                ax.text(x, -2.0, f"B{batch}", ha="center", va="top", fontsize=8, clip_on=False)
        ax.axhline(stats["mape_pct"], color="#333333", linestyle="--", linewidth=1, zorder=2)
        ax.set_ylim(0, limit)
        ax.set_xlim(-.55, 5.55)
        ax.set_yticks(np.linspace(0, limit, 4))
        ax.set_ylabel("Absolute percentage error (%)")
        ax.set_xticks(range(6), [f"{label}\nS={MATRIX[model][0]}" for label, model in zip(MODEL_LABELS, MATRIX)])
        ax.tick_params(axis="x", pad=26, length=0)
        style_axes(ax)
    fig.text(.075, .027, "Bars: APE against the three-repeat median. Whiskers: min-max repeat APE, not confidence intervals.\n"
             "B = batch size; S = sequence length. OOM has no measured APE and is excluded from MAPE.", fontsize=9)
    return fig


def plot_operators(operators):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.20, wspace=.23)
    fig.suptitle("RTX A6000: operator-category prediction error", fontsize=16, y=.98)
    fig.legend(handles=[Patch(facecolor=BLUE, label="Inference"),
                        Patch(facecolor=RED, hatch="//", label="Training")],
               loc="upper center", bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    for ax, field, title, ylabel, limit in zip(
        axes, ("mape_pct", "mean_absolute_error_ms"), ("(a) Category MAPE", "(b) Mean absolute category error"),
        ("Mean absolute percentage error (%)", "Mean absolute error (ms)"), (55, 255)
    ):
        limit = max(limit, operators[field].max() * 1.3)
        for rank, (mode, color) in enumerate(zip(MODES, (BLUE, RED))):
            selected = operators[operators["mode"] == mode].set_index("category").loc[list(CATEGORIES)]
            x = np.arange(5) + (-.19 if rank == 0 else .19)
            ax.bar(x, selected[field], width=.33, color=color, hatch="//" if rank else None,
                   edgecolor="white", linewidth=.6, zorder=3)
            for position, row in zip(x, selected.to_dict("records")):
                label = f"{row[field]:.1f}"
                if field == "mape_pct":
                    label += f"\nn={int(row['valid_count'])}"
                ax.text(position, row[field] + limit * .018, label, ha="center", va="bottom", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(5), CATEGORY_LABELS)
        ax.set_ylim(0, limit)
        style_axes(ax)
    fig.text(.075, .05, "Workload-level category totals, measured independently of full-model latency; three-repeat medians.\n"
             "FC = Linear; EW = VEC; LN = LayerNorm. Zero LN labels are excluded. These errors are not additive E2E residuals.",
             fontsize=9)
    return fig


def plot_common(common):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.075, right=.985, top=.74, bottom=.24, wspace=.22)
    fig.suptitle("NeuSight: historical GPUs + RTX A6000", fontsize=16, y=.98)
    fig.legend(handles=[Patch(facecolor=GRAY, label="Historical result"),
                        Patch(facecolor=GREEN, hatch="//", label="A6000 supplement (3-repeat median)")],
               loc="upper center", bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    limit = max(35, 5 * np.ceil(common["mape_pct"].max() * 1.25 / 5))
    for ax, mode, panel in zip(axes, MODES, ("a", "b")):
        rows = common[common["mode"] == mode].set_index("device")
        devices = [device for device in DEVICE_LABELS if device in rows.index]
        rows = rows.loc[devices]
        colors = [GREEN if device == "NVIDIA_RTX_A6000" else GRAY for device in devices]
        bars = ax.bar(range(len(rows)), rows["mape_pct"], width=.64, color=colors, edgecolor="white", zorder=3)
        bars[-1].set_hatch("//")
        for position, value in enumerate(rows["mape_pct"]):
            ax.text(position, value + .7, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        title = "Inference" if mode == "inf" else "Training"
        ax.set_title(f"({panel}) {title}: {int(rows.iloc[0]['valid_count'])} shared configurations", loc="left", pad=12, fontsize=11)
        ax.set_ylabel("Mean absolute percentage error (%)")
        ax.set_xticks(range(len(rows)), [DEVICE_LABELS[device] for device in devices], fontsize=8.5)
        ax.set_ylim(0, limit)
        ax.set_yticks(np.arange(0, limit + 1, max(10, 10 * np.ceil(limit / 40))))
        style_axes(ax)
    fig.text(.075, .06, "Identical workload sets within each panel; these subset MAPEs are not the full A6000 MAPEs.\n"
             "Historical software/TF32 metadata is incomplete: this is contextual comparison, not a matched-environment experiment.",
             fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY)
    parser.add_argument("--output-dir", type=Path, help="Defaults to <summary-dir>/figures")
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"), default=("png", "svg", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    require(args.dpi > 0, "DPI must be positive")
    directory = args.summary_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else directory / "figures"
    e2e, metrics, operators, common, sources = load_data(directory)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "neusight-a6000", "savefig.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=True)
    figures = {"a6000_e2e_error": plot_e2e(e2e, metrics),
               "a6000_operator_error": plot_operators(operators),
               "a6000_gpu_comparison": plot_common(common)}
    for name, figure in figures.items():
        for extension in args.formats:
            metadata = {"Creator": "NeuSight A6000 supplement"}
            if extension == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            elif extension == "svg":
                metadata["Date"] = None
            else:
                metadata = {"Software": "NeuSight A6000 supplement"}
            figure.savefig(output / f"{name}.{extension}", dpi=args.dpi, metadata=metadata)
        plt.close(figure)
    snapshot = output / "data"
    snapshot.mkdir(exist_ok=True)
    columns = ["name", "model", "mode", "seqlen", "batch", "status", "valid_repeats", "predicted_e2e_ms",
               "run_01_e2e_ms", "run_02_e2e_ms", "run_03_e2e_ms", "median_e2e_ms", "median_e2e_ape_pct"]
    for name, table in (("e2e.csv", e2e[columns]), ("metrics.csv", metrics),
                        ("operator_errors.csv", operators), ("common_gpu_errors.csv", common)):
        table.to_csv(snapshot / name, index=False)
    require(all(sha256(directory / name) == digest for name, digest in sources.items()),
            "Source summaries changed during plotting")
    write_json(output / "manifest.json", {"source_sha256": sources, "dpi": args.dpi,
               "formats": list(args.formats), "primary_statistic": "median of three independent repeats",
               "error_whiskers": "min-max repeat APE; not confidence intervals",
               "zero_and_failed_labels": "excluded, never zero-filled",
               "common_comparison": "all_gpu_common; identical configuration sets within each mode",
               "matplotlib_version": matplotlib.__version__,
               "output_sha256": {str(path.relative_to(output)): sha256(path)
                                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}})
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Generated {len(figures)} figures ({', '.join(args.formats)}) and data snapshots in {output}")


if __name__ == "__main__":
    main()
