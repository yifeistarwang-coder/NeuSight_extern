# Standard library imports
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import json

# Third party imports
import pandas as pd

# Constants
PREDICTOR_LIST = ["roofline", "habitat", "micro", "neusight"]
OOD_MODELS = ["gpt3_27", "opt_13", "gpt3_xl"]
OOD_DEVICES = ["NVIDIA_H100_80GB_HBM3", "NVIDIA_L4", "NVIDIA_A100_80GB_PCIe"]

# Operation types and modes
OPERATION_TYPES = ["all", "BMM", "Linear", "VEC", "VECsoftmax", "VECln"]
LATENCY_MODES = ["e2e", "fw", "bwall", "bw", "acc"]

# Model layer markers
MODEL_LAYERS = {
    "bert": {
        "start": "bert_encoder_layer_0_attention_self_query",
        "end": "bert_encoder_layer_0_output_layer_norm"
    },
    "gpt": {
        "start": "transformer_h_0_ln_1",
        "end": "add_15"
    },
    "opt": {
        "start": "model_decoder_layers_0_self_attn_layer_norm",
        "end": "view_11"
    }
}

def replicate_layer(trace, model_name, n_layer):
    if n_layer == 0:
        return trace

    if "switch" in model_name.lower():
        return trace

    for model_type, markers in MODEL_LAYERS.items():
        if model_type in model_name.lower():
            start = trace['Name'].loc[lambda x: x==markers["start"]].index.item()
            end = trace['Name'].loc[lambda x: x==markers["end"]].index.item() + 1
            break
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    begin = trace.iloc[:start]
    mid = trace.iloc[start:end]
    end = trace.iloc[end:]

    df = pd.concat([begin, *([mid]*n_layer), end])
    df = df.reset_index(drop=True)

    return df

def collect_perop_stat(entry, pred_dir, graph_dir, label_dir, device, model_desc, model_name, n_layer):
    dfs = []

    # Read label CSV
    label_csv_path = label_dir/device/(model_desc+".csv")
    if os.path.exists(label_csv_path):
        label_trace = pd.read_csv(label_csv_path)
        label_trace = label_trace.rename(columns={
            "latency": f"e2e_latency"
        })
        label_trace["bwall_latency"] = label_trace["bw_latency"] + label_trace["acc_latency"]

        label_trace = label_trace.rename(columns={
            tag+"_latency": f"label_{tag}_latency"
            for tag in LATENCY_MODES
        })
        
        # Merge with parsed graph
        graph_csv_path = graph_dir/(model_desc+".csv")
        graph_trace = pd.read_csv(graph_csv_path)
        label_trace = pd.merge(label_trace, graph_trace, how="left")
        dfs.append(("label", label_trace))

    # Collect predictions
    for predictor in PREDICTOR_LIST:
        pred_csv_path = pred_dir/device/predictor/(model_desc+".csv")
        if os.path.exists(pred_csv_path):
            df = pd.read_csv(pred_csv_path)
            df = df.rename(columns={
                tag+"_latency": f"{predictor}_{tag}_latency"
                for tag in LATENCY_MODES
            })
            dfs.append((predictor, df))



    # Collect per op stats
    for predictor, trace in dfs:
        trace = replicate_layer(trace, model_name, n_layer)
        per_op = trace.groupby(by="OpName").sum()
        
        # Drop unnecessary operations
        for op in ["contiguous", "dropout", "misc"]:
            per_op = per_op.drop(index=op, errors='ignore')

        # Process basic operations
        for opname in per_op.index:
            for mode in LATENCY_MODES:
                col = f"{predictor}_{mode}_latency"
                if col in per_op.loc[opname].index:
                    entry[f"{predictor}_{opname}_{mode}_latency"] = per_op.loc[opname][col]

        # Process vector operations
        vec_queries = {
            "VEC": "OpName.str.contains(r'VEC(?!softmax|ln$)')",
            "VECsoftmax": "OpName.str.match('VECsoftmax')",
            "VECln": "OpName.str.match('VECln')"
        }
        
        for vec_type, query in vec_queries.items():
            aggvec = per_op.query(query).sum(axis=0)
            for mode in LATENCY_MODES:
                col = f"{predictor}_{mode}_latency"
                if col in aggvec.index:
                    entry[f"{predictor}_{vec_type}_{mode}_latency"] = aggvec[col]

    return entry

def parse_entry(pred_dir, graph_dir, label_dir, device, model_desc):

    split = str(model_desc).split(".")[0].split("-")

    option = ""
    if len(split) == 5:
        model_name, mode, sequence_length, batch_size, option = split
    else:
        model_name, mode, sequence_length, batch_size = split

    # make entry
    entry = {"model_name":model_name, "seqlen":sequence_length, "batch":batch_size, "device":device, "mode":mode, "option":option}

    # read label json
    label_json_path = label_dir/device/(model_desc+".json")
    if os.path.exists(label_json_path):
        with open(label_json_path, 'r') as file:
            label_json = json.load(file)
        e2e_label = label_json["e2e_latency"]
        fw_label = label_json["fw_latency"]
        bwall_label = label_json["bwall_latency"]
        n_layer = int(label_json["num_layer"])

        entry[f"label_all_e2e_latency"] = e2e_label
        entry[f"label_all_fw_latency"] = fw_label
        entry[f"label_all_bwall_latency"] = bwall_label
    else:
        return None

    # read pred json
    for predictor in PREDICTOR_LIST:
        pred_json_path = pred_dir/device/predictor/(model_desc+".json")
        if os.path.exists(pred_json_path):
            with open(pred_json_path, 'r') as f:
                pred_json = json.load(f)
            entry[f"{predictor}_all_e2e_latency"] = pred_json["e2e_latency"]
            entry[f"{predictor}_all_fw_latency"] = pred_json["fw_latency"]
            entry[f"{predictor}_all_bw_latency"] = pred_json["bw_latency"]
            entry[f"{predictor}_all_bwall_latency"] = pred_json["bwall_latency"]
            entry[f"{predictor}_all_acc_latency"] = pred_json["acc_latency"]

    # per op
    if option == "" and not "AMD" in device:
        entry = collect_perop_stat(entry, pred_dir, graph_dir, label_dir, device, model_desc, model_name, n_layer)

    return entry

def is_ood(model_name, device):
    return model_name in OOD_MODELS or device in OOD_DEVICES

def ape(label, pred):
    return abs(label - pred) / label * 100

def make_summary(result_dir, label_dir):
    df_list = []
    pred_dir = result_dir/"prediction"
    
    graph_dir = result_dir/"opgraph"
    model_descs = []
    for subdir, dirs, files in os.walk(str(graph_dir)):
        for file in files:
            # Check if the file is a CSV
            if file.endswith(".csv"):
                model_descs.append(file.split(".")[0])

    for device in listdir(pred_dir):
        print(device)
        entries = []
        for model_desc in model_descs:
            entry = parse_entry(pred_dir, graph_dir, label_dir, device, model_desc)
            if entry is not None:
                entries.append(entry)

        df = pd.DataFrame(entries)
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.copy()

    # drop row if no label
    # df = df[df['label_all_e2e_latency'].notna()]

    # ape error
    for predictor in PREDICTOR_LIST:
        for ops in OPERATION_TYPES:
            for mode in ["e2e", "fw", "bwall"]:
                label_col = f"label_{ops}_{mode}_latency"
                pred_col = f"{predictor}_{ops}_{mode}_latency"
                err_col = f"{predictor}_{ops}_{mode}_err"
                
                df[err_col] = ape(df[label_col], df[pred_col])
    
    df = df.replace(float("inf"), 0)
    df["OOD"] = df.apply(lambda x: is_ood(x["model_name"], x["device"]), axis=1)

    return df

def main():
    result_dir = Path("./results")
    label_dir = Path("./label")
    summary_dir = Path("./summary")

    df = make_summary(result_dir, label_dir)
    df.to_csv(summary_dir/"summary.csv", index=False)

if __name__ == "__main__":
    main()
