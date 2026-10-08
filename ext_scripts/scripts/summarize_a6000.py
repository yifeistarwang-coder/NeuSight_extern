import math
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from a6000_common import (
    ASPLOS, DEVICE, LABELS, LOGS, RESULTS, ROOT, SUMMARY, output_directory,
    read_json, sha256, validate_outputs, verify_frozen, workloads, write_json,
)


PHASES = ("e2e", "fw", "bwall")
CATEGORIES = ("Linear", "BMM", "VEC", "VECsoftmax", "VECln")
OOD_MODELS = ("gpt3_xl", "gpt3_27", "opt_13")
OOD_DEVICES = ("NVIDIA_H100_80GB_HBM3", "NVIDIA_L4", "NVIDIA_A100_80GB_PCIe", DEVICE)


def ape(prediction, measured):
    if not math.isfinite(prediction) or not math.isfinite(measured) or measured <= 0:
        return float("nan")
    return abs(prediction - measured) / measured * 100


def operation_totals(table, model, n_layer):
    sys.path.insert(0, str(ROOT))
    from neusight.Prediction.aggregator import replicate_layer

    expanded = replicate_layer(table, model, n_layer)
    names = expanded["OpName"]
    selections = {
        "all": pd.Series(True, index=expanded.index),
        "Linear": names.eq("Linear"), "BMM": names.eq("BMM"),
        "VEC": names.str.contains(r"VEC(?!softmax|ln$)", regex=True),
        "VECsoftmax": names.str.match("VECsoftmax"), "VECln": names.str.match("VECln"),
    }
    return {category: {phase: float(expanded.loc[selected, f"{phase}_latency"].sum())
                       for phase in ("e2e", "fw", "bwall", "bw", "acc")}
            for category, selected in selections.items()}


def load_operations(label_csv, graph_csv, model, n_layer):
    label = pd.read_csv(label_csv)
    graph = pd.read_csv(graph_csv)
    if set(label["Name"]) != set(graph["Name"]):
        missing = sorted(set(label["Name"]) ^ set(graph["Name"]))
        raise ValueError(f"Label/graph node mismatch: {missing}")
    if label["Name"].duplicated().any() or graph["Name"].duplicated().any():
        raise ValueError("Non-unique graph nodes")
    label = label.merge(graph[["Name", "OpName"]], on="Name", how="left", validate="one_to_one")
    if label["OpName"].isna().any():
        raise ValueError("Unmatched label nodes")
    label["bwall_latency"] = label["bw_latency"] + label["acc_latency"]
    label["e2e_latency"] = label["fw_latency"] + label["bwall_latency"]
    return operation_totals(label, model, n_layer)


def distribution(values):
    values = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return {"valid_count": len(values), "mape_pct": float(values.mean()),
            "median_ape_pct": float(values.median()), "p90_ape_pct": float(values.quantile(.9))}


def summarize_workloads():
    rows, repeats, operator_rows, issues = [], [], [], []
    for workload in workloads():
        name = workload["name"]
        row = dict(workload)
        predictions = list((LOGS / "status/predict").glob(f"gpu*/run_00/{name}.json"))
        pred_records = [read_json(path) for path in predictions]
        valid_pred = [record for record in pred_records if record["status"] == "success"]
        if len(valid_pred) > 1:
            raise ValueError(f"Duplicate prediction: {name}")
        prediction = None
        predicted_operations = None
        if valid_pred:
            prediction = validate_outputs("predict", RESULTS / "prediction" / DEVICE / "neusight", name)
            pred_table = pd.read_csv(RESULTS / "prediction" / DEVICE / "neusight" / f"{name}.csv")
            predicted_operations = operation_totals(pred_table, workload["model"], prediction["num_layer"])
        row["prediction_status"] = "success" if valid_pred else (
            pred_records[-1]["status"] if pred_records else "missing")
        for phase in PHASES:
            row[f"predicted_{phase}_ms"] = prediction[f"{phase}_latency"] if prediction else np.nan
        measurements = []
        terminal = []
        durations = []
        reasons = []
        for run in (1, 2, 3):
            candidates = list((LOGS / "status/formal").glob(f"gpu*/run_{run:02d}/{name}.json"))
            if len(candidates) > 1:
                raise ValueError(f"Duplicate formal run: {name}/{run}")
            record = read_json(candidates[0]) if candidates else {"status": "missing"}
            if "physical_gpu" in record:
                row["physical_gpu"] = record["physical_gpu"]
            if "elapsed_seconds" in record:
                durations.append(record["elapsed_seconds"])
            if record.get("error"):
                reasons.append(record["error"])
            terminal.append(record["status"])
            repeat = {**workload, "run": run, "status": record["status"],
                      "physical_gpu": record.get("physical_gpu"), "error": record.get("error", ""),
                      "elapsed_seconds": record.get("elapsed_seconds"), "log_path": record.get("log_path", "")}
            row[f"run_{run:02d}_status"] = record["status"]
            row[f"run_{run:02d}_seconds"] = record.get("elapsed_seconds", np.nan)
            if record["status"] == "success":
                directory = ROOT / record["output_directory"]
                label = validate_outputs("formal", directory, name)
                for suffix, digest in record["output_hashes"].items():
                    if sha256(directory / f"{name}.{suffix}") != digest:
                        raise ValueError(f"Modified raw output: {name}/{run}/{suffix}")
                measurements.append(label)
                measured_operations = None
                graph_csv = RESULTS / "opgraph" / f"{name}.csv"
                if graph_csv.exists():
                    measured_operations = load_operations(directory / f"{name}.csv", graph_csv,
                                                          workload["model"], label["num_layer"])
                for phase in PHASES:
                    measured = label[f"{phase}_latency"]
                    predicted = row[f"predicted_{phase}_ms"]
                    row[f"run_{run:02d}_{phase}_ms"] = measured
                    row[f"run_{run:02d}_{phase}_ape_pct"] = ape(predicted, measured)
                    repeat[f"measured_{phase}_ms"] = measured
                    repeat[f"ape_{phase}_pct"] = ape(predicted, measured)
                if measured_operations:
                    for category, values in measured_operations.items():
                        for phase in PHASES:
                            measured = values[phase]
                            predicted = predicted_operations[category][phase] if predicted_operations else np.nan
                            operator_rows.append({**workload, "physical_gpu": record["physical_gpu"],
                                                  "run": run, "category": category, "phase": phase,
                                                  "predicted_ms": predicted, "measured_ms": measured,
                                                  "ape_pct": ape(predicted, measured),
                                                  "validity": "valid" if measured > 0 and math.isfinite(predicted)
                                                  else "zero_label" if measured == 0 else "missing_prediction"})
            else:
                issues.append({**workload, "run": run, "status": record["status"],
                               "physical_gpu": record.get("physical_gpu"),
                               "error": record.get("error", ""), "log_path": record.get("log_path", "")})
            repeats.append(repeat)
        row["valid_repeats"] = len(measurements)
        row["sum_repeat_seconds"] = sum(durations) if len(durations) == 3 else np.nan
        row["failure_reasons"] = "; ".join(dict.fromkeys(reasons))
        row["status"] = "success" if len(measurements) == 3 and prediction else (
            "oom" if terminal == ["oom"] * 3 else "partial" if measurements else terminal[-1])
        for phase in PHASES:
            measured = [entry[f"{phase}_latency"] for entry in measurements]
            median = float(np.median(measured)) if len(measured) == 3 else np.nan
            row[f"median_{phase}_ms"] = median
            row[f"min_{phase}_ms"] = min(measured) if measured else np.nan
            row[f"max_{phase}_ms"] = max(measured) if measured else np.nan
            row[f"median_{phase}_ape_pct"] = ape(row[f"predicted_{phase}_ms"], median)
            row[f"predicted_over_measured_{phase}"] = row[f"predicted_{phase}_ms"] / median if median > 0 else np.nan
            row[f"range_over_median_{phase}_pct"] = (
                (max(measured) - min(measured)) / median * 100 if median > 0 else np.nan)
        rows.append(row)
    return (pd.DataFrame(rows), pd.DataFrame(repeats), pd.DataFrame(operator_rows),
            pd.DataFrame(issues, columns=("model", "mode", "seqlen", "batch", "name", "run", "status",
                                          "physical_gpu", "error", "log_path")))


def summarize_operators(operators):
    rows = []
    if operators.empty:
        return pd.DataFrame(), pd.DataFrame()
    for keys, group in operators.groupby(["name", "model", "mode", "category", "phase"], sort=False):
        row = dict(zip(("name", "model", "mode", "category", "phase"), keys))
        measured = group["measured_ms"]
        median = float(measured.median()) if len(group) == 3 else np.nan
        prediction = float(group["predicted_ms"].iloc[0])
        row.update(valid_repeats=len(group), predicted_ms=prediction, median_measured_ms=median,
                   min_measured_ms=float(measured.min()), max_measured_ms=float(measured.max()),
                   ape_pct=ape(prediction, median), model_group="OOD" if row["model"] in OOD_MODELS else "ID",
                   signed_error_ms=prediction - median, absolute_error_ms=abs(prediction - median),
                   device_group="OOD", group="OOD", validity="valid" if median > 0 else "zero_or_incomplete_label")
        rows.append(row)
    medians = pd.DataFrame(rows)
    aggregate = []
    for keys, group in medians.groupby(["mode", "model_group", "category", "phase"], sort=False):
        aggregate.append({**dict(zip(("mode", "model_group", "category", "phase"), keys)),
                          "group": "OOD", "group_scope": "model_stratum",
                          "total_count": len(group), "excluded_count": int(group["ape_pct"].isna().sum()),
                          **distribution(group["ape_pct"])})
    for keys, group in medians.groupby(["mode", "category", "phase"], sort=False):
        valid = group[group["ape_pct"].notna()]
        aggregate.append({"mode": keys[0], "model_group": "mixed", "category": keys[1], "phase": keys[2],
                          "group": "OOD", "group_scope": "all_models",
                          "mean_signed_error_ms": float(valid["signed_error_ms"].mean()),
                          "mean_absolute_error_ms": float(valid["absolute_error_ms"].mean()),
                          "total_count": len(group), "excluded_count": int(group["ape_pct"].isna().sum()),
                          **distribution(group["ape_pct"])})
    for keys, group in medians.groupby(["category", "phase"], sort=False):
        aggregate.append({"mode": "mixed", "model_group": "mixed", "category": keys[0], "phase": keys[1],
                          "group": "OOD", "group_scope": "legacy_mixed",
                          "total_count": len(group), "excluded_count": int(group["ape_pct"].isna().sum()),
                          **distribution(group["ape_pct"])})
    return medians, pd.DataFrame(aggregate)


def smoke_tables():
    rows = []
    for path in sorted((LOGS / "status/smoke").glob("gpu*/run_00/*.json")):
        record = read_json(path)
        row = {key: record.get(key) for key in ("name", "model", "mode", "physical_gpu", "status", "elapsed_seconds", "error")}
        row.update(record.get("latency", {}))
        rows.append(row)
    smoke = pd.DataFrame(rows)
    differences = []
    if not smoke.empty:
        for name, group in smoke[smoke["status"] == "success"].groupby("name"):
            if len(group) == 2:
                group = group.sort_values("physical_gpu")
                first, second = group.iloc[0], group.iloc[1]
                differences.append({"name": name, "gpu_a": int(first["physical_gpu"]),
                                    "gpu_b": int(second["physical_gpu"]),
                                    "gpu_a_e2e_ms": first["e2e_latency"], "gpu_b_e2e_ms": second["e2e_latency"],
                                    "relative_difference_pct": abs(first["e2e_latency"] - second["e2e_latency"]) /
                                    ((first["e2e_latency"] + second["e2e_latency"]) / 2) * 100})
    return smoke, pd.DataFrame(differences)


def legacy_tables(current):
    old = pd.read_csv(ASPLOS / "summary/summary.csv")
    old = old[~old["device"].str.contains("AMD") & old["option"].isna()].copy()
    legacy_rows, comparisons = [], []
    valid_current = current[current["status"] == "success"].set_index("name")
    by_device = {}
    for entry in old.to_dict("records"):
        name = f"{entry['model_name']}-{entry['mode']}-{int(entry['seqlen'])}-{int(entry['batch'])}"
        if name not in {item["name"] for item in workloads()}:
            continue
        device = entry["device"]
        label_path = ASPLOS / "label" / device / f"{name}.json"
        pred_path = ASPLOS / "results/prediction" / device / "neusight" / f"{name}.json"
        if not label_path.exists() or not pred_path.exists():
            continue
        label, pred = read_json(label_path), read_json(pred_path)
        by_device.setdefault(device, {})[name] = ape(pred["e2e_latency"], label["e2e_latency"])
        label_operations = load_operations(label_path.with_suffix(".csv"),
                                           ASPLOS / "results/opgraph" / f"{name}.csv",
                                           entry["model_name"], label["num_layer"])
        pred_operations = operation_totals(pd.read_csv(pred_path.with_suffix(".csv")),
                                           entry["model_name"], pred["num_layer"])
        for category in CATEGORIES:
            for phase in PHASES:
                measured, predicted = label_operations[category][phase], pred_operations[category][phase]
                legacy_rows.append({"device": device, "name": name, "model": entry["model_name"],
                                    "mode": entry["mode"], "category": category, "phase": phase,
                                    "corrected_measured_ms": measured, "predicted_ms": predicted,
                                    "ape_pct": ape(predicted, measured),
                                    "group": "OOD" if device in OOD_DEVICES or entry["model_name"] in OOD_MODELS else "ID"})
    mode_names = {mode: {item["name"] for item in workloads() if item["mode"] == mode}
                  for mode in ("inf", "train")}
    global_common = {}
    for mode, names in mode_names.items():
        global_common[mode] = set(valid_current.index) & names
        for entries in by_device.values():
            if set(entries) & names:
                global_common[mode].intersection_update(entries)
    details = []
    for device, entries in by_device.items():
        pair_common = set(valid_current.index) & set(entries)
        for scope in ("pairwise_common", "all_gpu_common"):
            for mode in ("inf", "train"):
                if not set(entries) & mode_names[mode]:
                    continue
                common = pair_common if scope == "pairwise_common" else global_common[mode]
                names = sorted(name for name in common if valid_current.loc[name, "mode"] == mode)
                comparisons.append({"reference_device": device, "mode": mode, "scope": scope,
                                    "valid_count": len(names), "configurations": ";".join(names),
                                    "a6000_mape_pct": float(valid_current.loc[names, "median_e2e_ape_pct"].mean()),
                                    "reference_mape_pct": float(np.mean([entries[name] for name in names])) if names else np.nan})
                for name in names:
                    details.append({"reference_device": device, "scope": scope, "mode": mode, "name": name,
                                    "a6000_ape_pct": valid_current.loc[name, "median_e2e_ape_pct"],
                                    "reference_ape_pct": entries[name]})
    legacy = pd.DataFrame(legacy_rows)
    means = []
    if not legacy.empty:
        for keys, group in legacy[legacy["phase"] == "e2e"].groupby(["group", "category"]):
            means.append({"group": keys[0], "category": keys[1], "total_count": len(group),
                          "excluded_count": int(group["ape_pct"].isna().sum()), **distribution(group["ape_pct"])})
    return legacy, pd.DataFrame(comparisons), pd.DataFrame(details), pd.DataFrame(means)


def summarize():
    verify_frozen()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    current, repeats, operators, issues = summarize_workloads()
    medians, category_metrics = summarize_operators(operators)
    smoke, differences = smoke_tables()
    legacy, comparisons, comparison_details, legacy_means = legacy_tables(current)
    metrics = []
    for mode in ("inf", "train"):
        group = current[current["mode"] == mode]
        for phase in PHASES:
            metrics.append({"mode": mode, "phase": phase, "total_count": len(group),
                            "excluded_count": int(group[f"median_{phase}_ape_pct"].isna().sum()),
                            **distribution(group[f"median_{phase}_ape_pct"])})
    metric_table = pd.DataFrame(metrics)
    history_records = [read_json(path) for path in (LOGS / "status/formal").glob("gpu*/run_*/history/*.json")]
    history_columns = ("name", "model", "mode", "physical_gpu", "run", "status", "error", "elapsed_seconds", "log_path")
    history_table = pd.DataFrame([{key: record.get(key) for key in history_columns} for record in history_records],
                                 columns=history_columns)
    timing_rows = []
    for kind in ("predict", "smoke", "formal"):
        records = [read_json(path) for path in (LOGS / "status" / kind).glob("gpu*/run_*/*.json")]
        if kind == "formal":
            records += history_records
        records = [record for record in records if "finished_at" in record]
        if not records:
            continue
        for gpu in ("all", *sorted({record["physical_gpu"] for record in records})):
            selected = records if gpu == "all" else [record for record in records if record["physical_gpu"] == gpu]
            first = min(datetime.fromisoformat(record["started_at"]) for record in selected)
            last = max(datetime.fromisoformat(record["finished_at"]) for record in selected)
            timing_rows.append({"phase": kind, "physical_gpu": gpu, "attempts": len(selected),
                                "successful_attempts": sum(record["status"] == "success" for record in selected),
                                "first_started_at": first.isoformat(), "last_finished_at": last.isoformat(),
                                "span_seconds": (last - first).total_seconds(),
                                "sum_attempt_seconds": sum(record["elapsed_seconds"] for record in selected)})
    timing = pd.DataFrame(timing_rows)
    tables = {
        "e2e.csv": current, "repeats.csv": repeats, "metrics.csv": metric_table,
        "training_decomposition.csv": current[current["mode"] == "train"],
        "operator_repeats.csv": operators, "operator_medians.csv": medians,
        "operator_metrics.csv": category_metrics, "failures.csv": issues,
        "attempt_history.csv": history_table,
        "execution_timing.csv": timing,
        "smoke.csv": smoke, "gpu_comparison.csv": differences,
        "legacy_operator_decomposition_corrected.csv": legacy,
        "common_config_comparison.csv": comparisons, "common_config_details.csv": comparison_details,
        "legacy_category_means.csv": legacy_means,
    }
    for filename, table in tables.items():
        table.to_csv(SUMMARY / filename, index=False)
    completed = len(repeats) == 72 and not repeats["status"].isin(("missing", "running")).any()
    prediction_count = sum(current["prediction_status"] == "success")
    audit = {"formal_attempts": len(repeats), "terminal_attempts": int((~repeats["status"].isin(("missing", "running"))).sum()),
             "successful_attempts": int((repeats["status"] == "success").sum()),
             "successful_configurations": int((current["status"] == "success").sum()),
             "archived_attempts": len(history_records), "total_attempts_including_retries": len(repeats) + len(history_records),
             "predictions": int(prediction_count), "all_formal_attempts_terminal": bool(completed),
             "frozen_inputs_unchanged": True, "gpu_comparison_pairs": len(differences),
             "output_tables": {filename: {"rows": len(table), "sha256": sha256(SUMMARY / filename)}
                               for filename, table in tables.items()}}
    write_json(SUMMARY / "audit.json", audit)
    report = ["# A6000 experiment results", "",
              "Single GPU, FP32, eager, matmul/cuDNN TF32 disabled; frozen predictor and original tile tables.",
              "Each formal repeat uses 25 samples; average of fastest five forward/backward separately. Training excludes loss forward and optimizer.",
              "Primary statistics use complete three-repeat medians only; failed, incomplete and zero labels are excluded, never zero-filled.", "",
              f"Formal attempts: {audit['terminal_attempts']}/72 terminal; {audit['successful_attempts']} successful; complete configurations: {audit['successful_configurations']}/24.",
              f"Archived failed attempts before retries: {len(history_records)}; total attempts including retries: {audit['total_attempts_including_retries']}. Archived attempts are excluded from metrics and preserved in attempt_history.csv and status history JSON files.",
              f"Predictions: {prediction_count}/24 successful; frozen inputs unchanged.", "",
              "| Mode | Valid | MAPE (%) | Median APE (%) | P90 APE (%) |",
              "|---|---:|---:|---:|---:|"]
    for entry in metrics:
        if entry["phase"] == "e2e":
            report.append(f"| {entry['mode']} | {entry['valid_count']} | {entry['mape_pct']:.3f} | {entry['median_ape_pct']:.3f} | {entry['p90_ape_pct']:.3f} |")
    report.extend(["", "## Operator error localization", "",
                   "Category MAPE is the mean workload-level category APE, not a mean of individual operator instances. Absolute errors use the same positive-label valid samples.",
                   "| Mode | Category | Valid | MAPE (%) | Mean Absolute Error (ms) | Mean Signed Error (ms) |",
                   "|---|---|---:|---:|---:|---:|"])
    if not category_metrics.empty:
        categories = category_metrics[(category_metrics["phase"] == "e2e") &
                                      (category_metrics["group_scope"] == "all_models") &
                                      category_metrics["category"].isin(CATEGORIES)]
        for entry in categories.to_dict("records"):
            report.append(f"| {entry['mode']} | {entry['category']} | {entry['valid_count']} | {entry['mape_pct']:.3f} | {entry['mean_absolute_error_ms']:.3f} | {entry['mean_signed_error_ms']:.3f} |")
        report.append("")
        for mode in ("inf", "train"):
            group = categories[(categories["mode"] == mode) & (categories["valid_count"] > 0)]
            if not group.empty:
                relative = group.loc[group["mape_pct"].idxmax()]
                absolute = group.loc[group["mean_absolute_error_ms"].idxmax()]
                report.append(f"For {mode}, {relative['category']} has the largest category MAPE ({relative['mape_pct']:.3f}%); {absolute['category']} has the largest mean absolute category error ({absolute['mean_absolute_error_ms']:.3f} ms).")
        report.append("These category discrepancies are descriptive, not an additive decomposition of the end-to-end residual, because operator and full-model timings are separate measurements.")
    report.extend(["", "## Final statuses", "",
                   "| Status | Configurations |", "|---|---:|"])
    for status, count in current["status"].value_counts().items():
        report.append(f"| {status} | {count} |")
    report.extend(["", "## Execution timing", "",
                   "| Phase | Attempts | Time Span (minutes) | Summed Attempt Time (minutes) |",
                   "|---|---:|---:|---:|"])
    for entry in timing[timing["physical_gpu"] == "all"].to_dict("records"):
        report.append(f"| {entry['phase']} | {entry['attempts']} | {entry['span_seconds'] / 60:.2f} | {entry['sum_attempt_seconds'] / 60:.2f} |")
    report.append("Time span includes gaps/interleaving between phase invocations; summed attempt time is not wall time because the two GPUs run concurrently.")
    report.extend(["", "## Measurement stability", "",
                   f"Maximum formal end-to-end range/median: {current['range_over_median_e2e_pct'].max():.3f}%."])
    if not differences.empty:
        report.append(f"Four paired pilot workloads: inter-GPU absolute difference / pair mean ranges from {differences['relative_difference_pct'].min():.3f}% to {differences['relative_difference_pct'].max():.3f}%.")
    report.extend(["", "## Interpretation and limitations", "",
                   "A6000 is an unseen GPU model, not an unseen architecture: A100 is present in frozen training tables.",
                   f"Training accuracy covers only {int(((current['mode'] == 'train') & (current['status'] == 'success')).sum())}/12 configurations that completed FP32 measurement. OOM configurations retain predictions and failure records but cannot contribute measured APE under this protocol.",
                   "Operator sums are measured on the single-layer graph and expanded using the repository layer markers (Switch uses its full graph); they are not end-to-end model timing.",
                   "Switch label num_layer=0 is the repository convention for disabling layer replication; its full encoder/decoder graph is still measured.",
                   "Forward/backward and category medians are computed independently; sums of medians need not equal the median of summed latencies.",
                   "Operator VEC/EW filtering and ID/OOD grouping follow the existing analysis. Old operator backward totals are recomputed as bw + acc in the corrected local table.",
                   "Old results lack full environment and TF32 records. Cross-GPU MAPE uses explicitly listed common configurations only.",
                   "Synthetic token inputs and random model initialization do not represent autoregressive decode or production serving.", "",
                   "## Artifacts", "",
                   "Raw labels: scripts/asplos/label_a6000; predictions and graphs: scripts/asplos/results_a6000.",
                   "Environment, hashes, per-attempt statuses/logs and GPU telemetry: scripts/asplos/logs_a6000.",
                   "All tables and this report remain local, excluded from Git."])
    (SUMMARY / "REPORT.md").write_text("\n".join(report) + "\n")
    print(metric_table.to_string(index=False))
    print(audit)


if __name__ == "__main__":
    summarize()
