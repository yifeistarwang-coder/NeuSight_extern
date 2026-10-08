import argparse
import concurrent.futures
import csv
import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from a6000_common import (
    ASPLOS, LOGS, ROOT, frozen_inputs, read_json, status_path,
    verify_frozen, workloads, write_json,
)


WORKER = Path(__file__).with_name("a6000_worker.py")
SMOKE = ("bert_large-inf-512-8", "bert_large-train-512-2",
         "gpt2_large-inf-1024-4", "gpt2_large-train-1024-1")


def gpu_inventory():
    fields = "index,uuid,pci.bus_id,name,memory.used,memory.total,utilization.gpu,power.limit,ecc.mode.current"
    text = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
    return [dict(zip(fields.split(","), (value.strip() for value in row)))
            for row in csv.reader(text.splitlines())]


def compute_processes():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True)
    return list(csv.reader(text.splitlines()))


def wait_idle(gpu):
    for attempt in range(150):
        inventory = next(item for item in gpu_inventory() if int(item["index"]) == gpu)
        processes = [row for row in compute_processes() if row[0].strip() == inventory["uuid"]]
        if not processes and int(inventory["memory.used"]) < 256 and int(inventory["utilization.gpu"]) == 0:
            return inventory
        if attempt % 15 == 0:
            print(f"GPU {gpu} occupied; waiting without touching other processes", flush=True)
        time.sleep(2)
    raise RuntimeError(f"GPU {gpu} did not become idle; no measurement launched")


def run_job(kind, gpu, name=None, run=0):
    if name:
        saved = status_path(kind, gpu, run, name)
        if saved.exists() and read_json(saved)["status"] == "success":
            from a6000_common import output_directory, validate_outputs
            validate_outputs(kind, output_directory(kind, gpu, run), name)
            print(f"Keep {kind}/gpu{gpu}/run{run}: {name}", flush=True)
            return read_json(saved)
    gpu_record = wait_idle(gpu)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu_record["uuid"],
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
               PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tag = f"{kind}_gpu{gpu}_run{run:02d}_{name or 'environment'}"
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = LOGS / f"{tag}_{stamp}.log"
    telemetry_path = LOGS / f"{tag}_{stamp}_telemetry.csv"
    processes_path = LOGS / f"{tag}_{stamp}_processes.csv"
    command = [sys.executable, str(WORKER), kind, "--gpu", str(gpu), "--run", str(run)]
    if name:
        command.extend(["--name", name])
    print(f"Start {tag}; log={log_path.relative_to(ROOT)}", flush=True)
    with log_path.open("w") as log, telemetry_path.open("w") as telemetry, processes_path.open("w") as processes:
        monitor = subprocess.Popen(
            ["nvidia-smi", "-i", str(gpu),
             "--query-gpu=timestamp,index,uuid,pci.bus_id,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit",
             "--format=csv", "-l", "2"], stdout=telemetry, stderr=telemetry)
        process_monitor = subprocess.Popen(
            ["nvidia-smi", "--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory",
             "--format=csv", "-l", "2"], stdout=processes, stderr=processes)
        try:
            worker = subprocess.Popen(command, cwd=ASPLOS, env=env, stdout=log, stderr=subprocess.STDOUT)
            returncode = worker.wait()
        finally:
            monitor.terminate()
            monitor.wait()
            process_monitor.terminate()
            process_monitor.wait()
    if name:
        if not saved.exists():
            raise RuntimeError(f"Worker did not write status (exit {returncode}): {log_path}")
        record = read_json(saved)
        record.update(log_path=str(log_path.relative_to(ROOT)),
                      telemetry_path=str(telemetry_path.relative_to(ROOT)),
                      processes_path=str(processes_path.relative_to(ROOT)), worker_pid=worker.pid,
                      gpu_inventory_before=gpu_record, processes_after=compute_processes(),
                      worker_command=command, worker_returncode=returncode)
        write_json(saved, record)
        print(f"Done {tag}: {record['status']} ({record.get('elapsed_seconds', 0):.1f}s)", flush=True)
        return record
    if returncode:
        raise RuntimeError(f"Environment probe failed: {log_path}")


def parallel_lanes(gpus, action):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(action, gpu) for gpu in gpus]
        return [future.result() for future in futures]


def prepare(gpus):
    inputs = frozen_inputs()
    manifest = LOGS / "frozen_inputs.json"
    if manifest.exists():
        verify_frozen()
    else:
        write_json(manifest, inputs)
    snapshot = LOGS / "preparation.json"
    if not snapshot.exists():
        write_json(snapshot, {"created_at": datetime.datetime.now().astimezone().isoformat(),
                              "gpus": gpu_inventory(), "selected_gpus": gpus,
                              "compute_processes": compute_processes(), "matrix": workloads(),
                              "git_commit": subprocess.check_output(
                                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        (LOGS / "topology.txt").write_text(subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True))
        (LOGS / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        (LOGS / "source_worktree.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True))
    parallel_lanes(gpus, lambda gpu: run_job("environment", gpu))


def predict(gpus, names):
    selected = [item for item in workloads() if not names or item["name"] in names]
    from a6000_common import output_directory, validate_outputs
    existing = [read_json(path) for path in (LOGS / "status/predict").glob("gpu*/run_00/*.json")]
    completed = {record["name"] for record in existing if record["status"] == "success"}
    for item in selected:
        if item["name"] in completed:
            validate_outputs("predict", output_directory("predict", gpus[0], 0), item["name"])
    selected = [item for item in selected if item["name"] not in completed]
    print(f"Generating {len(selected)} predictions; preserving validated existing outputs", flush=True)
    partitions = {gpu: selected[index::len(gpus)] for index, gpu in enumerate(gpus)}
    return parallel_lanes(gpus, lambda gpu: [run_job("predict", gpu, item["name"])
                                            for item in partitions[gpu]])


def smoke(gpus):
    parallel_lanes(gpus, lambda gpu: [run_job("smoke", gpu, name) for name in SMOKE])
    extra = ("gpt3_27-inf-2048-2", "switchxl4-inf-512-1")
    parallel_lanes(gpus, lambda gpu: run_job("smoke", gpu, extra[gpus.index(gpu)]))


def formal(gpus, names=None):
    from a6000_common import output_directory, validate_outputs
    prediction_records = [read_json(path) for path in (LOGS / "status/predict").glob("gpu*/run_00/*.json")]
    if len(prediction_records) != 24 or any(record["status"] in ("missing", "running") for record in prediction_records):
        raise RuntimeError("Generate all 24 predictions before formal measurement")
    pilot_records = [read_json(path) for path in (LOGS / "status/smoke").glob("gpu*/run_00/*.json")]
    if len(pilot_records) != 10 or any(record["status"] == "running" for record in pilot_records):
        raise RuntimeError("Finish the eight base and two additional pilot attempts first")
    for gpu in gpus:
        for name in SMOKE:
            status_file = status_path("smoke", gpu, 0, name)
            if not status_file.exists() or read_json(status_file)["status"] != "success":
                raise RuntimeError(f"Pilot gate not satisfied: GPU {gpu}, {name}")
            validate_outputs("smoke", output_directory("smoke", gpu, 0), name)
            validate_outputs("predict", output_directory("predict", gpu, 0), name)
    from summarize_a6000 import smoke_tables
    smoke_summary, gpu_differences = smoke_tables()
    if len(gpu_differences) != 4:
        raise RuntimeError("Missing paired inter-GPU pilot comparison")
    (ASPLOS / "summary_a6000").mkdir(parents=True, exist_ok=True)
    smoke_summary.to_csv(ASPLOS / "summary_a6000/smoke.csv", index=False)
    gpu_differences.to_csv(ASPLOS / "summary_a6000/gpu_comparison.csv", index=False)
    print(gpu_differences.to_string(index=False), flush=True)
    path = LOGS / "assignments.json"
    if path.exists():
        assignments = read_json(path)
        if sorted(map(int, assignments)) != sorted(gpus):
            raise RuntimeError("Existing assignment uses different GPUs")
    else:
        costs = {}
        probes = [record for record in pilot_records if record["status"] == "success"]
        for item in workloads():
            matches = [record for record in probes if record["name"] == item["name"]]
            if matches:
                costs[item["name"]] = sum(record["elapsed_seconds"] for record in matches) / len(matches)
                continue
            related = [record for record in probes if record["model"] == item["model"]]
            if not related:
                related = [record for record in probes if record["model"] == "gpt2_large"
                           and record["mode"] == item["mode"]]
            reference = related[0]
            ratio = item["batch"] * item["seqlen"] / (reference["batch"] * reference["seqlen"])
            scale = .7 + .3 * ratio if item["model"] == "switchxl4" else max(.5, ratio)
            if item["mode"] != reference["mode"]:
                scale *= 1.5 if item["mode"] == "train" else 1 / 1.5
            costs[item["name"]] = reference["elapsed_seconds"] * scale
        assignments = {str(gpu): [] for gpu in gpus}
        loads = {str(gpu): 0 for gpu in gpus}
        for name in sorted(costs, key=costs.get, reverse=True):
            candidates = [key for key in loads if len(assignments[key]) < 12]
            chosen = min(candidates, key=loads.get)
            assignments[chosen].append(name)
            loads[chosen] += costs[name]
        write_json(path, assignments)
        write_json(LOGS / "cost_estimates.json", {"estimated_seconds_per_configuration": costs,
                                                  "estimated_formal_seconds_per_gpu": {
                                                      key: value * 3 for key, value in loads.items()},
                                                  "note": "Scheduling estimates from pilot wall time, not measured results"})
    selected_names = {item["name"] for item in workloads()} if not names else set(names)
    if not selected_names <= {item["name"] for item in workloads()}:
        raise ValueError("Unknown formal retry workload")
    parallel_lanes(gpus, lambda gpu: [run_job("formal", gpu, name, run)
                                    for name in assignments[str(gpu)] if name in selected_names
                                    for run in (1, 2, 3)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "predict", "smoke", "formal", "summary"))
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 2])
    parser.add_argument("--names", nargs="*")
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        raise ValueError("Select two distinct idle physical GPUs")
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "prepare":
            prepare(args.gpus)
        else:
            verify_frozen()
            if args.phase == "predict":
                predict(args.gpus, args.names)
            elif args.phase == "smoke":
                smoke(args.gpus)
            elif args.phase == "formal":
                formal(args.gpus, args.names)
            else:
                from summarize_a6000 import summarize
                summarize()


if __name__ == "__main__":
    main()
