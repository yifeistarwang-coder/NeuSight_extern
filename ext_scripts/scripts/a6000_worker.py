import argparse
import ctypes
import datetime
import importlib.metadata
import os
import random
import subprocess
import sys
import time
import traceback

from a6000_common import (
    ASPLOS, DEVICE, LOGS, ROOT, output_directory, read_json, sha256,
    status_path, validate_outputs, verify_frozen, workloads, write_json,
)


def environment():
    import torch

    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "packages": {package: importlib.metadata.version(package) for package in
                     ("torch", "torchvision", "transformers", "numpy", "pandas", "scipy", "tensorboard")},
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "default_dtype": str(torch.get_default_dtype()),
        "attention": "eager",
        "seed": 42,
        "timing": {"iterations": 25, "fastest_samples": 5,
                   "training": "forward + backward; excludes loss forward and optimizer"},
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "logical_device": "cuda:0",
        "gpu_name": properties.name,
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "torch_threads": torch.get_num_threads(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path)
                          for pattern in ("neusight/**/*.py", "ext_scripts/scripts/*.py")
                          for path in sorted(ROOT.glob(pattern))},
        "frozen_manifest_sha256": sha256(LOGS / "frozen_inputs.json"),
    }


def ensure_driver_library():
    try:
        ctypes.CDLL("libcuda.so")
    except OSError:
        candidates = ("/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib/x86_64-linux-gnu/libcuda.so.1")
        from pathlib import Path
        source = next((Path(path).resolve() for path in candidates if Path(path).exists()), None)
        if source is None:
            raise RuntimeError("Cannot locate the installed 64-bit CUDA driver library")
        directory = ROOT / ".venv_a6000/driver_libs"
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / "libcuda.so"
        if not link.exists():
            try:
                link.symlink_to(source)
            except FileExistsError:
                pass
        if str(directory) in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            raise RuntimeError("Local CUDA driver alias did not resolve libcuda.so")
        library_path = ":".join(value for value in (str(directory), os.environ.get("LD_LIBRARY_PATH", "")) if value)
        env = dict(os.environ, LD_LIBRARY_PATH=library_path)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def main():
    ensure_driver_library()
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("predict", "smoke", "formal", "environment"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run", type=int, default=0)
    args = parser.parse_args()
    verify_frozen()

    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    details = environment()
    if details["gpu_name"] != "NVIDIA RTX A6000" or details["sm_count"] != 84:
        raise RuntimeError("Unexpected GPU mapping")
    if args.kind == "environment":
        write_json(LOGS / f"environment_gpu{args.gpu}.json", details)
        print(json_format(details), flush=True)
        return

    workload = next(item for item in workloads() if item["name"] == args.name)
    destination = output_directory(args.kind, args.gpu, args.run)
    status_file = status_path(args.kind, args.gpu, args.run, args.name)
    if status_file.exists() and read_json(status_file)["status"] == "success":
        validate_outputs(args.kind, destination, args.name)
        print(f"Preserving valid result: {args.name}", flush=True)
        return
    if status_file.exists():
        previous = read_json(status_file)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        write_json(status_file.parent / "history" / f"{args.name}_{stamp}.json", previous)
    if any((destination / f"{args.name}.{suffix}").exists() for suffix in ("csv", "json")):
        raise RuntimeError(f"Existing unvalidated output must be inspected: {destination / args.name}")
    started = time.monotonic()
    record = {
        "kind": args.kind, "physical_gpu": args.gpu, "run": args.run,
        **workload, "environment": details,
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "running", "output_directory": str(destination.relative_to(ROOT)),
    }
    write_json(status_file, record)
    try:
        config = ASPLOS / "data/DLmodel_configs" / f"{workload['model']}.json"
        if args.kind == "predict":
            from neusight import NeusightPredictor
            predictor = NeusightPredictor(
                predictor_name="neusight", predictor_path=ASPLOS / "data/predictor/MLP_WAVE",
                tile_dataset_dir=ASPLOS / "data/dataset/train",
            )
            predictor.predict(
                device_config_path=ASPLOS / "data/device_configs" / f"{DEVICE}.json",
                model_config_path=config, sequence_length=workload["seqlen"],
                batch_size=workload["batch"], execution_type=workload["mode"],
                result_dir=ASPLOS / "results_a6000",
            )
        else:
            from neusight.Tracing.trace import trace_graph
            table, entry = trace_graph(
                model_config_path=config, sequence_length=workload["seqlen"],
                batch_size=workload["batch"], is_train=workload["mode"] == "train",
                bench=True, single_layer="switch" not in workload["model"],
            )
            if table is None or entry is None:
                raise torch.cuda.OutOfMemoryError("End-to-end measurement returned out of memory")
            destination.mkdir(parents=True, exist_ok=True)
            table.to_csv(destination / f"{args.name}.csv", index=False)
            write_json(destination / f"{args.name}.json", entry)
        entry = validate_outputs(args.kind, destination, args.name)
        record.update(status="success", latency=entry,
                      output_hashes={suffix: sha256(destination / f"{args.name}.{suffix}")
                                     for suffix in ("csv", "json")})
    except Exception as error:
        message = str(error)
        if "out of memory" in message.lower():
            failure = "oom"
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            failure = "dependency_error"
        elif isinstance(error, (FileNotFoundError, ValueError)):
            failure = "output_or_parse_error"
        else:
            failure = "trace_or_runtime_error"
        record.update(status=failure, error=message, traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        write_json(status_file, record)
    print(f"{args.kind} {args.name}: {record['status']} ({record['elapsed_seconds']:.1f}s)", flush=True)
    if record["status"] != "success":
        sys.exit(2)


def json_format(value):
    import json
    return json.dumps(value, indent=2)


if __name__ == "__main__":
    main()
