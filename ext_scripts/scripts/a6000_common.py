import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASPLOS = ROOT / "scripts/asplos"
LOGS = ASPLOS / "logs_a6000"
RESULTS = ASPLOS / "results_a6000"
LABELS = ASPLOS / "label_a6000"
SUMMARY = ASPLOS / "summary_a6000"
DEVICE = "NVIDIA_RTX_A6000"
MATRIX = {
    "bert_large": (512, (8, 16), (2, 8)),
    "gpt2_large": (1024, (4, 8), (1, 4)),
    "gpt3_xl": (2048, (2, 8), (1, 2)),
    "opt_13": (2048, (2, 8), (1, 2)),
    "gpt3_27": (2048, (2, 8), (1, 2)),
    "switchxl4": (512, (1, 2), (1, 2)),
}


def workloads():
    return [
        {"model": model, "mode": mode, "seqlen": seqlen, "batch": batch,
         "name": f"{model}-{mode}-{seqlen}-{batch}"}
        for model, (seqlen, inference, training) in MATRIX.items()
        for mode, batches in (("inf", inference), ("train", training))
        for batch in batches
    ]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_inputs():
    data = ASPLOS / "data"
    paths = [
        data / "predictor/MLP_WAVE" / operation / filename
        for operation in ("LINEAR", "BMM", "VEC", "SOFTMAX", "LN")
        for filename in ("model.pth", "config.json")
    ]
    paths.extend(data / "dataset/train" / f"{operation}.csv"
                 for operation in ("linear", "bmm", "vec"))
    paths.extend(sorted((data / "dataset/train/device_configs").glob("*.json")))
    paths.extend(data / "DLmodel_configs" / f"{model}.json" for model in MATRIX)
    paths.append(data / "device_configs" / f"{DEVICE}.json")
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def verify_frozen():
    expected = read_json(LOGS / "frozen_inputs.json")
    actual = frozen_inputs()
    if expected != actual:
        changed = sorted(path for path in expected.keys() | actual.keys()
                         if expected.get(path) != actual.get(path))
        raise RuntimeError(f"Frozen inputs changed: {changed}")
    return actual


def output_directory(kind, gpu, run):
    if kind == "predict":
        return RESULTS / "prediction" / DEVICE / "neusight"
    if kind == "smoke":
        return LABELS / "smoke" / f"gpu{gpu}" / DEVICE
    return LABELS / f"gpu{gpu}" / f"run_{run:02d}" / DEVICE


def status_path(kind, gpu, run, name):
    return LOGS / "status" / kind / f"gpu{gpu}" / f"run_{run:02d}" / f"{name}.json"


def validate_outputs(kind, directory, name):
    import math
    import pandas as pd

    table = pd.read_csv(directory / f"{name}.csv")
    entry = read_json(directory / f"{name}.json")
    if table.empty or table["Name"].duplicated().any():
        raise ValueError("Empty operator trace or duplicate node names")
    fields = ("e2e_latency", "fw_latency", "bwall_latency")
    if any(not math.isfinite(entry[field]) or entry[field] < 0 for field in fields):
        raise ValueError("Invalid end-to-end latency")
    if entry["e2e_latency"] <= 0:
        raise ValueError("Non-positive end-to-end label")
    if not math.isclose(entry["e2e_latency"], entry["fw_latency"] + entry["bwall_latency"],
                        rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Forward/backward do not sum to end-to-end")
    columns = ("fw_latency", "bw_latency", "acc_latency")
    if table[list(columns)].isna().any().any():
        raise ValueError("Missing operator latency")
    if not table[list(columns)].map(lambda value: math.isfinite(value) and value >= 0).all().all():
        raise ValueError("Invalid operator latency")
    dtype_table = pd.read_csv(RESULTS / "opgraph_raw" / f"{name}.csv") if kind == "predict" else table
    floating_dtypes = dtype_table["dtype"].dropna().astype(str)
    floating_dtypes = floating_dtypes[floating_dtypes.str.contains("float|half", regex=True)]
    if set(floating_dtypes) != {"torch.float32"}:
        raise ValueError(f"Unexpected floating-point types: {sorted(set(floating_dtypes))}")
    if kind == "predict":
        for subdirectory in ("opgraph", "opgraph_raw"):
            graph = pd.read_csv(RESULTS / subdirectory / f"{name}.csv")
            if graph.empty or graph["Name"].tolist() != table["Name"].tolist():
                raise ValueError(f"{subdirectory} and prediction nodes do not correspond")
    return entry
