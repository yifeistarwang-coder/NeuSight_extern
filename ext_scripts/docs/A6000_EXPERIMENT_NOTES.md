# A6000 实验执行附录

方案 review 见 [A6000_EXPERIMENT_PLAN.md](A6000_EXPERIMENT_PLAN.md)。本附录保存原规划的硬件依据、命令和实现待办；其中“待新增／尚未生成”等表述是规划时的状态，不代表当前执行状态。本轮设备配置、runner、试跑、正式采集和汇总已完成，重新核查的环境、命令及本地结果路径见 [AGNETS.md](AGNETS.md) 和 [A6000_RUNNER.md](A6000_RUNNER.md)。以下历史硬件与环境信息仅作参考。

除 Markdown 链接外，下文相对路径均以 `scripts/asplos/` 为基准。命令需在该目录执行。

## 1. 依据与冻结输入

论文：*Forecasting GPU Performance for Deep Learning Training and Inference*，ASPLOS 2025，DOI：`10.1145/3669940.3707265`。

| 内容 | 依据 |
|---|---|
| 推理／训练端到端误差 | [Figure 7(a)/(b)](https://arxiv.org/html/2407.13853v3#S6.F7) |
| BMM、FC、EW、Softmax、LN 类别误差 | [Figure 8](https://arxiv.org/html/2407.13853v3#S6.F8) |
| 模型规格 | [Table 5](https://arxiv.org/html/2407.13853v3#S6.T5) |
| 本次模型、序列长度、batch size | [预测脚本](scripts/asplos/run_pred_nvidia_neusight.py)、[采集脚本](scripts/asplos/run_label_nvidia.py) |

Figure 7 包含四种预测器；本次仅补充 NeuSight 的 A6000 结果。三轮独立重复是本次增加的稳定性检查。

冻结以下输入，并保存文件哈希：

- `data/predictor/MLP_WAVE/{LINEAR,BMM,VEC,SOFTMAX,LN}/` 下的 `model.pth` 和 `config.json`。
- `data/dataset/train/{linear,bmm,vec}.csv` 及 `data/dataset/train/device_configs/`。
- `data/DLmodel_configs/` 中本次使用的六个模型配置，以及待新增的 A6000 设备 JSON。

原规划记录：五组权重和配置均存在；上述三个训练 CSV 包含 P100、V100、P4、T4、A100，没有 A6000。预测同时依赖权重和 tile 表，A6000 测量数据必须独立保存，不能回填这些表或用于调参。

被测模型由 Hugging Face 配置随机初始化，输入为合成数据，无需下载它们的预训练权重。被冻结的预训练权重属于 NeuSight 预测器。

## 2. 硬件记录与设备 JSON

原规划检查时，本机有 8 张 NVIDIA RTX A6000，各卡下列属性一致；实验只选两张空闲卡。

| 项目 | 原核查结果 |
|---|---|
| Compute Capability／SM 数量 | 8.6／84 |
| L2／驱动可见显存 | 6 MiB（6,291,456 bytes）／49140 MiB |
| Driver API 核心时钟属性 | 1,800,000 kHz |
| 显存时钟属性／位宽 | 8,001,000 kHz／384 bit |
| 功耗限制／ECC | 300 W／Disabled |
| 驱动版本 | 580.178.04 |

待新增：`data/device_configs/NVIDIA_RTX_A6000.json`。

```json
{
  "Device": "NVIDIA RTX A6000",
  "Dev_Mem": 48,
  "Mem_Bw": 768,
  "Num_Sm": 84,
  "Core_Per_SM": 128,
  "Freq": 1800,
  "SingleFLOPs": 38707,
  "L2Cache": 6
}
```

字段口径：

- 五个预测器读取 `Num_Sm`、`SingleFLOPs`、`Dev_Mem`、`Mem_Bw`、`L2Cache`；`Device` 用于标识，`Core_Per_SM` 和 `Freq` 仅保留为规格信息。
- `SingleFLOPs` 为理论 FP32 GFLOPS：`84 × 128 × 2 × 1.8 = 38707.2`，约 38.7 TFLOPS。`Mem_Bw` 为标称 768 GB/s，两者均不是实测值。
- `Dev_Mem = 48` 使用标称 GB，`L2Cache = 6` 沿用仓库容量口径。独立单卡任务始终使用每卡 84 个 SM、48 GB 显存。
- `Freq = 1800` 沿用现有 JSON 的 MHz 惯例；README 的单位说明与现有文件不同。原核查中 `nvidia-smi` 最大支持 SM 时钟为 2100 MHz，与 Driver API 属性含义不同，不据此替换主实验理论峰值。
- 张量维度、`MemPerO`、`OpsPerO` 等输入由算子图提供，无需新增硬件字段。不得用实测有效算力或带宽拟合 A6000 标签。

原拓扑记录：GPU 0–3 属于 NUMA 0，GPU 4–7 属于 NUMA 1。优先选同一 NUMA 节点的空闲卡，GPU 0 和 2 仅为候选。运行前重新检查占用，记录物理编号、UUID、PCI Bus ID、NUMA，以及 `CUDA_VISIBLE_DEVICES` 到进程内 `cuda:0` 的映射。

## 3. 环境与计时实现

优先使用 PyTorch 2.1.0、Transformers 4.38.1，依据分别为 [README](README.md) 和 [setup.py](setup.py)。`cu121` 是本次拟用的 CUDA 构建，仓库未保存旧结果的 CUDA/cuDNN 版本，不能视为已核实的原环境。README 测试环境为 Python 3.9；原规划检查的系统 Python 为 3.10.12，未安装 torch、transformers、pandas、numpy。先查找已有兼容环境，再决定是否新建独立环境。

运行记录应包含：依赖版本（含 CUDA runtime、cuDNN、NumPy、pandas）、FP32/eager 设置、matmul 与 cuDNN 的 TF32 开关（均关闭）、随机种子、仓库提交与代码变更、输入文件哈希。另记录 GPU 驱动、功耗限制、ECC，以及测量期间的时钟、温度、功耗和其他任务占用。

计时依据：[trace.py](neusight/Tracing/trace.py)、[analysis.py](neusight/Tracing/analysis.py)。

- 每轮计时主体执行 25 次，前向、反向分别排序并取最快 5 次均值，再相加得到端到端延迟；不是先将每次前反向相加再选最快 5 次。三轮指独立启动三次采集流程，是本次新增。
- 训练 JSON 的 `e2e_latency` 为前向加反向，字段分别为 `fw_latency`、`bwall_latency`；不包含 optimizer step，loss 计算在这两个计时区间外。
- 推理测给定序列的一次 forward，不包含逐 token 自回归或 KV-cache decode。
- 普通 Transformer 默认 trace 一层，汇总时按层数展开；端到端标签测完整模型。Switch 的 trace 路径不同，需单独检查。
- 算子时间之和与完整模型端到端时间分别报告。原始 CSV 的 `bw_latency`、`acc_latency` 分别记录反向计算和梯度累积，完整算子反向应取二者之和；现有汇总代码误用减法，需修正并重算用于对照的旧算子反向分解。端到端主指标直接对照 JSON，不受该问题影响。

## 4. 试跑、调度与输出隔离

两张卡各运行以下四项，共八次试跑：

| 配置名 | 模式 | 序列长度 | batch size |
|---|---|---:|---:|
| `bert_large` | `inf` | 512 | 8 |
| `bert_large` | `train` | 512 | 2 |
| `gpt2_large` | `inf` | 1024 | 4 |
| `gpt2_large` | `train` | 1024 | 1 |

检查预测、标签、算子图、汇总与卡间差异，再补跑一个较大 GPT 配置和一个 Switch 配置，校准预算。正式任务按耗时均衡分配，每卡约 12 个配置、36 次采集；同一配置的三轮尽量使用同一卡。每卡同时只跑一个采集进程，预测与正式测量分阶段执行。

建议产物目录（均待生成）：

```text
scripts/asplos/
├── data/device_configs/NVIDIA_RTX_A6000.json
├── results_a6000/
│   ├── opgraph_raw/
│   ├── opgraph/
│   └── prediction/NVIDIA_RTX_A6000/neusight/
├── label_a6000/
│   ├── smoke/gpu{物理编号}/...
│   └── gpu{物理编号}/run_{01,02,03}/NVIDIA_RTX_A6000/...
├── logs_a6000/
└── summary_a6000/
```

采集脚本按 GPU 型号命名目录，两张 A6000 会重名，必须用不同 `result_dir` 隔离物理卡和轮次。保留原始 CSV／JSON；试跑与补充配置单列，不混入正式三轮。

## 5. 单配置命令示例

环境和设备 JSON 就绪后，从仓库根目录进入执行目录。以下使用候选物理 GPU 0。

预测：

```bash
cd scripts/asplos

CUDA_VISIBLE_DEVICES=0 python3 ../pred.py \
  --predictor_name neusight \
  --predictor_path ./data/predictor/MLP_WAVE \
  --device_config_path ./data/device_configs/NVIDIA_RTX_A6000.json \
  --model_config_path ./data/DLmodel_configs/bert_large.json \
  --sequence_length 512 \
  --batch_size 8 \
  --execution_type inf \
  --tile_dataset_dir ./data/dataset/train \
  --result_dir ./results_a6000
```

标签采集（同一工作目录）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 ./collect_label.py \
  --model_config_path ./data/DLmodel_configs/bert_large.json \
  --mode inf \
  --batch_size 8 \
  --seqlen 512 \
  --result_dir ./label_a6000/gpu0/run_01 \
  --overwrite false
```

第二张卡更换物理编号和输出目录，每轮使用新的 `run_XX`。上述命令仅展示接口；完整 runner 仍需落实环境设置和输出校验。现有 `--overwrite false` 主要检查 CSV 是否存在，续跑必须同时验证 CSV、JSON 完整且有效。

## 6. 实现待办与报表规则

| 位置 | 待完成内容 |
|---|---|
| A6000 device JSON | 写入第 2 节硬件特征 |
| 新增 runner | 统一矩阵、预测／采集阶段、两卡调度、三轮重复、日志、耗时、失败记录、有效输出校验及续跑 |
| runner 预检 | 检查五组权重完整且成功加载；现有加载逻辑可能在缺少 `model.pth` 时继续使用初始化模型 |
| 路径处理 | 避免沿用 `run_label_nvidia.py` 写死的 `~/NeuSight/...` 路径，改用脚本位置或显式参数 |
| `summarize.py` 或独立脚本 | 支持自定义目录、仅 NeuSight、分卡分轮、缺失项和三轮统计；修正算子 `bwall = bw + acc`，取消将无穷误差置零，并统一重算参与对照的旧数据 |
| `table.py` 或独立脚本 | 处理 GPU 列表和 OOD 判断的硬编码，加入 A6000，输出推理／训练／算子表 |

无需修改五个预测器的结构或输入维度。

报表规则：

- 每项保留三轮实测和各轮 APE，用于对照旧结果；主表另外列三轮中位数、对应 APE、最小／最大值及预测／实测比值。推理、训练分别汇总 MAPE、中位 APE、P90 APE 和有效配置数；跨 GPU 平均误差比较另取共同有效配置。
- 算子类别先在每个 workload 内汇总耗时，再计算类别误差与跨 workload 的平均值。与旧类别结果对照时统一算子筛选、层数展开、ID/OOD 分组及有效样本；旧 `perop.csv` 混合推理和训练，新实验可另列两种模式。逐算子实例误差另行命名。
- 零值、缺失标签、OOM 和失败保留状态，不作为零误差计入平均；记录排除原因和每项耗时。缺少有效重复的配置需标明，不冒充完整三轮。
- 区分 OOM、trace／parse 错误、依赖问题和输出不完整。失败可续跑，已完成的有效轮次不覆盖；降低 batch 的补充测试单列。

预算明细（原规划估计，未经试跑）：环境与配置 30–120 分钟，双卡试跑 20–60 分钟，预测与算子图 30–90 分钟，正式采集 3–9 小时，汇总 30–60 分钟。正式采集按 `24 × 3 × 单项耗时 / 2 = 36 × 单项耗时` 估算；每项 5／10／15 分钟分别对应 3／6／9 小时。依赖问题、额外试跑、Switch trace 或重跑可能增加总时间。

## 7. 后续扩展的前提

若某类算子误差较大，可针对实际 workload shape 补采微基准，检查 kernel／tile 选择差异；诊断数据独立保存。使用 A6000 tile 信息的实验应单独作为消融实验。

分布式实验需另立方案：原拓扑检查仅显示 `NODE`／`SYS`，未显示 NVLink；现有通信预测只对特定 H100／A100 设置 `link_bw`，A6000 进入 `None` 分支。增加 DDP／TP／PP 前需确认 P2P 与拓扑、补充通信模型，并处理现有逻辑中的四卡假设。

## 8. 与现有实验数据的一致性核查

核查对象为仓库中已保存的 [label](scripts/asplos/label)、[summary.csv](scripts/asplos/summary/summary.csv)、[perop.csv](scripts/asplos/summary/perop.csv) 及当前采集／汇总代码；本次未运行 GPU 实验。当前源码可以说明实现口径，但旧结果缺少完整运行元数据，无法证明其环境和源码版本完全相同。

| 检查项 | 结论与依据 |
|---|---|
| 配置矩阵 | 24 项与预测、采集脚本一致。现存 NVIDIA 非融合、非分布式主结果共 80 行，覆盖其中 22 项；`gpt3_xl-train-2048-2`、`gpt3_27-train-2048-2` 没有旧主结果，缺失原因未保存。 |
| 精度与执行方式 | 抽查 H100 的 BERT 推理／训练、Switch 训练 CSV，浮点张量为 `torch.float32`；当前 `get_model()` 使用 eager attention，普通路径不调用 `torch.compile`。 |
| TF32 与环境 | 当前 `trace.py` 显式关闭 cuDNN TF32，未显式设置 matmul TF32；149 份旧标签 JSON 均只有层数和三个延迟字段，没有 TF32、CUDA/cuDNN、驱动、种子或轮次记录。新实验显式关闭两种 TF32 并记录环境，历史设置仍不可确认。 |
| 端到端定义与 APE | 80 行旧主结果的端到端／前向／反向标签均与原 JSON 一致，NeuSight 端到端 APE 均可由预测和标签 JSON 重算吻合。训练计时为分别采样后的前向 + 反向，loss 的前向计算和 optimizer step 不计入。 |
| 重复与新增指标 | 旧结果每个设备／配置只保存一组汇总标签，没有三轮记录。三轮中位数、测量波动、中位 APE、P90 和按 GPU／模式汇总的 MAPE 是本次额外输出；保留每轮结果供旧口径对照。 |
| 算子类别均值 | `table_perop()` 先筛选 NVIDIA 普通执行结果，再按 ID/OOD 对每个 workload 的类别 APE 求算术均值，混合 GPU、推理和训练。按此规则重算的 10 个 NeuSight 均值与已保存的 `perop.csv` 一致；A6000 单卡型号结果属于新的样本集合。 |

已确认的汇总问题与处理边界：

- [analysis.py](neusight/Tracing/analysis.py) 将算子耗时定义为 `fw + bw + acc`，而 [summarize.py](scripts/asplos/summarize.py) 使用 `bwall = bw - acc`。旧 H100／BERT-Large／训练 batch 2 的 LayerNorm 类别 `bwall` 因而为 **−0.10624 ms**，正确的 `bw + acc` 为 **2.28673 ms**。修正后需同步重算旧算子反向分解；该错误不影响 JSON 端到端指标，也不影响使用 `e2e` 的旧类别 APE。
- 现有 `summarize.py` 将无穷误差改成 0，与本计划排除无效标签的策略不同。已保存主结果中的 10 个零标签类别项均为 LayerNorm 的 `0/0` 空误差，旧类别平均会跳过它们；新汇总应明确记录无效项和样本数。
- 旧主结果各 GPU 的成功配置数不同，例如 H100 为推理 12 项／训练 10 项，L4 为 7／3 项。不能直接用各自不同配置集合的总体 MAPE 判断 GPU 间精度差异；对照表需使用共同有效配置，完整 A6000 结果另列。
