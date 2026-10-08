import ast
import csv
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from a6000_common import (
    ASPLOS, DEVICE, LOGS, RESULTS, ROOT, SUMMARY, output_directory, read_json,
    sha256, validate_outputs, verify_frozen, workloads, write_json,
)


def check_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8), (actual, expected)


def verify_environment(record):
    environment = record["environment"]
    inventory = record["gpu_inventory_before"]
    assert int(inventory["index"]) == record["physical_gpu"]
    assert int(inventory["memory.used"]) < 256 and int(inventory["utilization.gpu"]) == 0
    assert float(inventory["power.limit"]) == 300 and inventory["ecc.mode.current"] == "Disabled"
    assert environment["packages"]["torch"].split("+")[0] == "2.1.0"
    assert environment["packages"]["transformers"] == "4.38.1"
    assert environment["cuda_runtime"] == "12.1"
    assert environment["default_dtype"] == "torch.float32"
    assert environment["attention"] == "eager"
    assert environment["seed"] == 42
    assert environment["matmul_tf32"] is False and environment["cudnn_tf32"] is False
    assert environment["sm_count"] == 84 and environment["compute_capability"] == [8, 6]
    assert 47 < environment["memory_bytes"] / (1024 ** 3) <= 48
    assert environment["gpu_name"] == "NVIDIA RTX A6000"
    assert environment["logical_device"] == "cuda:0"
    assert environment["cuda_visible_devices"] == record["gpu_inventory_before"]["uuid"]
    assert environment["frozen_manifest_sha256"] == sha256(LOGS / "frozen_inputs.json")
    assert environment["timing"]["iterations"] == 25 and environment["timing"]["fastest_samples"] == 5


def verify_process_samples(record):
    uuid = record["environment"]["cuda_visible_devices"]
    with (ROOT / record["processes_path"]).open() as source:
        rows = list(csv.reader(source))
    interference = [row for row in rows if len(row) == 5 and row[1].strip() == uuid
                    and row[2].strip() != str(record["worker_pid"])]
    assert not interference, (record["name"], interference)
    assert (ROOT / record["telemetry_path"]).stat().st_size > 0


def verify_raw_operator_totals(record, operators):
    name = record["name"]
    label = pd.read_csv(ROOT / record["output_directory"] / f"{name}.csv")
    graph = pd.read_csv(RESULTS / "opgraph" / f"{name}.csv")
    assert set(label["Name"]) == set(graph["Name"])
    label = label.merge(graph[["Name", "OpName"]], on="Name", validate="one_to_one", how="left")
    weights = np.ones(len(label))
    model = record["model"]
    if "switch" not in model:
        if "bert" in model:
            start_name, end_name = ("bert_encoder_layer_0_attention_self_query", "bert_encoder_layer_0_output_layer_norm")
        elif "gpt" in model:
            start_name = "transformer_h_0_ln_1_grad" if "transformer_h_0_ln_1_grad" in set(label["Name"]) else "transformer_h_0_ln_1"
            end_name = "add_15"
        else:
            start_name, end_name = "model_decoder_layers_0_self_attn_layer_norm", "view_11"
        start = label.index[label["Name"] == start_name].item()
        end = label.index[label["Name"] == end_name].item()
        assert start <= end
        weights[start:end + 1] = record["latency"]["num_layer"]
    names = label["OpName"]
    selections = {"all": np.ones(len(label), dtype=bool), "Linear": names.eq("Linear"), "BMM": names.eq("BMM"),
                  "VEC": names.str.contains(r"VEC(?!softmax|ln$)", regex=True),
                  "VECsoftmax": names.str.match("VECsoftmax"), "VECln": names.str.match("VECln")}
    backward = label["bw_latency"] + label["acc_latency"]
    phases = {"fw": label["fw_latency"], "bwall": backward, "e2e": label["fw_latency"] + backward}
    subset = operators[(operators["name"] == name) & (operators["run"] == record["run"])].set_index(["category", "phase"])
    assert subset.index.is_unique
    for category, selected in selections.items():
        for phase, values in phases.items():
            total = float(np.dot(weights[selected], values[selected]))
            check_close(subset.loc[(category, phase), "measured_ms"], total)


def verify():
    manifest = verify_frozen()
    original_paths = [path for path in manifest if path != f"scripts/asplos/data/device_configs/{DEVICE}.json"]
    original_paths.extend(str(path.relative_to(ROOT)) for path in ROOT.glob("neusight/**/*.py"))
    for path in original_paths:
        baseline = subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip()
        actual = subprocess.check_output(["git", "hash-object", path], cwd=ROOT, text=True).strip()
        assert baseline == actual, f"Changed original predictor input or benchmark source: {path}"
    expected = {item["name"] for item in workloads()}
    source = ast.parse((ASPLOS / "run_label_nvidia.py").read_text())
    original = set()
    for node in ast.walk(source):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "launch_task":
            args = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
            original.add(f"{args['model_name']}-{args['mode']}-{args['seqlen']}-{args['batch_size']}")
    assert original == expected and len(expected) == 24
    hardware = read_json(LOGS / "hardware_driver_attributes.json")
    configuration = read_json(ASPLOS / "data/device_configs" / f"{DEVICE}.json")
    for gpu in hardware["gpus"]:
        assert gpu["sm_count"] == configuration["Num_Sm"]
        assert gpu["l2_cache_bytes"] == configuration["L2Cache"] * 1024 * 1024
        assert gpu["core_clock_khz"] == configuration["Freq"] * 1000
        theoretical = gpu["sm_count"] * configuration["Core_Per_SM"] * 2 * gpu["core_clock_khz"] / 1e6
        assert abs(theoretical - configuration["SingleFLOPs"]) < 1
        bandwidth = gpu["memory_clock_khz"] * 2 * gpu["memory_bus_width_bits"] / 8 / 1e6
        assert math.isclose(bandwidth, configuration["Mem_Bw"], rel_tol=.001)
    predictions = [read_json(path) for path in (LOGS / "status/predict").glob("gpu*/run_00/*.json")]
    assert len(predictions) == 24 and {record["name"] for record in predictions} == expected
    assert all(record["status"] == "success" for record in predictions)
    formal = [read_json(path) for path in (LOGS / "status/formal").glob("gpu*/run_*/*.json")]
    assert len(formal) == 72
    assert {(record["name"], record["run"]) for record in formal} == {
        (name, run) for name in expected for run in (1, 2, 3)}
    smoke = [read_json(path) for path in (LOGS / "status/smoke").glob("gpu*/run_00/*.json")]
    assert len(smoke) == 10
    terminal = ("success", "oom", "trace_or_runtime_error", "output_or_parse_error", "dependency_error")
    for record in predictions + formal + smoke:
        assert record["status"] in terminal
        assert record["elapsed_seconds"] > 0
        verify_environment(record)
        verify_process_samples(record)
        assert (ROOT / record["log_path"]).exists()
        if record["status"] == "success":
            directory = ROOT / record["output_directory"]
            validate_outputs(record["kind"], directory, record["name"])
            for suffix, digest in record["output_hashes"].items():
                assert sha256(directory / f"{record['name']}.{suffix}") == digest
            if record["kind"] == "smoke":
                label_names = pd.read_csv(directory / f"{record['name']}.csv")["Name"]
                graph_names = pd.read_csv(RESULTS / "opgraph" / f"{record['name']}.csv")["Name"]
                assert set(label_names) == set(graph_names)
        else:
            assert record["error"] and record["traceback"]
    e2e = pd.read_csv(SUMMARY / "e2e.csv").set_index("name")
    repeats = pd.read_csv(SUMMARY / "repeats.csv")
    metrics = pd.read_csv(SUMMARY / "metrics.csv")
    assert set(e2e.index) == expected and len(repeats) == 72
    assert len(pd.read_csv(SUMMARY / "gpu_comparison.csv")) == 4
    for name in expected:
        records = [record for record in formal if record["name"] == name]
        assert len({record["physical_gpu"] for record in records}) == 1
        assert e2e.loc[name, "physical_gpu"] == records[0]["physical_gpu"]
        check_close(e2e.loc[name, "sum_repeat_seconds"], sum(record["elapsed_seconds"] for record in records))
        valid = [record for record in records if record["status"] == "success"]
        assert e2e.loc[name, "valid_repeats"] == len(valid)
        for phase in ("e2e", "fw", "bwall"):
            values = []
            for record in valid:
                label = read_json(ROOT / record["output_directory"] / f"{name}.json")
                value = label[f"{phase}_latency"]
                values.append(value)
                check_close(e2e.loc[name, f"run_{record['run']:02d}_{phase}_ms"], value)
                if value > 0:
                    error = abs(e2e.loc[name, f"predicted_{phase}_ms"] - value) / value * 100
                    check_close(e2e.loc[name, f"run_{record['run']:02d}_{phase}_ape_pct"], error)
            if len(values) == 3:
                median = float(np.median(values))
                check_close(e2e.loc[name, f"median_{phase}_ms"], median)
                check_close(e2e.loc[name, f"min_{phase}_ms"], min(values))
                check_close(e2e.loc[name, f"max_{phase}_ms"], max(values))
                if median > 0:
                    error = abs(e2e.loc[name, f"predicted_{phase}_ms"] - median) / median * 100
                    check_close(e2e.loc[name, f"median_{phase}_ape_pct"], error)
            else:
                assert pd.isna(e2e.loc[name, f"median_{phase}_ape_pct"])
    for entry in metrics.to_dict("records"):
        values = e2e.loc[e2e["mode"] == entry["mode"], f"median_{entry['phase']}_ape_pct"].dropna()
        assert entry["valid_count"] == len(values)
        if len(values):
            check_close(entry["mape_pct"], values.mean())
            check_close(entry["median_ape_pct"], values.median())
            check_close(entry["p90_ape_pct"], values.quantile(.9))
    successful = {record["name"] for record in formal if record["status"] == "success"}
    operators = pd.read_csv(SUMMARY / "operator_repeats.csv")
    assert set(operators["name"]) == successful
    assert set(operators["category"]) == {"all", "Linear", "BMM", "VEC", "VECsoftmax", "VECln"}
    assert len(operators) == sum(record["status"] == "success" for record in formal) * 6 * 3
    for record in formal:
        if record["status"] == "success":
            verify_raw_operator_totals(record, operators)
    positive = operators[operators["measured_ms"] > 0]
    differences = (positive["predicted_ms"] - positive["measured_ms"]).abs() / positive["measured_ms"] * 100
    assert np.allclose(positive["ape_pct"], differences, atol=1e-8)
    assert operators.loc[operators["measured_ms"] == 0, "ape_pct"].isna().all()
    comparisons = pd.read_csv(SUMMARY / "common_config_comparison.csv")
    for entry in comparisons.to_dict("records"):
        names = entry["configurations"].split(";") if isinstance(entry["configurations"], str) else []
        assert len(names) == entry["valid_count"]
        assert set(names) <= set(e2e[e2e["status"] == "success"].index)
        if names:
            check_close(entry["a6000_mape_pct"], e2e.loc[names, "median_e2e_ape_pct"].mean())
    audit = read_json(SUMMARY / "audit.json")
    history = [read_json(path) for path in (LOGS / "status/formal").glob("gpu*/run_*/history/*.json")]
    assert audit["archived_attempts"] == len(history)
    assert audit["total_attempts_including_retries"] == 72 + len(history)
    for record in history:
        assert record["status"] != "success" and record["error"]
        assert (ROOT / record["log_path"]).exists()
        verify_environment(record)
        verify_process_samples(record)
    assert audit["all_formal_attempts_terminal"] and audit["frozen_inputs_unchanged"]
    for filename, record in audit["output_tables"].items():
        assert sha256(SUMMARY / filename) == record["sha256"]
    verification = {"passed": True, "matrix_count": 24, "prediction_count": 24,
                    "formal_attempts": 72, "pilot_attempts": 10,
                    "successful_formal_attempts": sum(record["status"] == "success" for record in formal),
                    "frozen_inputs_unchanged": True, "no_sampled_gpu_interference": True,
                    "original_predictor_inputs_and_benchmark_sources_match_git": True,
                    "hardware_attributes_and_idle_gpu_preflight_checked": True,
                    "raw_operator_totals_independently_rederived": True,
                    "raw_artifact_hashes_checked": True, "independent_numeric_checks": True}
    write_json(SUMMARY / "verification.json", verification)
    print(verification)


if __name__ == "__main__":
    verify()
