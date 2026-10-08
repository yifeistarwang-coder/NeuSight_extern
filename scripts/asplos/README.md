# Experiment Workflow

The `results` and `summary` directories already contain the numbers shown in Figures 7, 8, and 9, and Tables 7 and 8 of the paper. Data for these figures and tables is available in `/scripts/asplos/summary`. Note that latency predictions may vary slightly (~10%) due to the non-deterministic behavior of DNN models used in the publication.

## RTX A6000 实验（中文）

### 实验范围与运行入口

本补充实验冻结现有预测器权重、原始 tile 表及模型输入配置，不重新训练、调参或将 A6000 实测数据回填训练集。以下说明与后文的原论文复现步骤分开使用。

- 每项任务使用单张 RTX A6000；两张空闲卡并行执行独立任务，不进行双卡模型并行。
- 固定 FP32、eager 和随机种子 42，关闭 matmul 与 cuDNN TF32。
- 从仓库根目录按[运行说明](../../ext_scripts/docs/A6000_RUNNER.md)依次执行环境准备、预测、试跑、正式测量、汇总与验收；完整配置矩阵见[实验计划](../../ext_scripts/docs/A6000_EXPERIMENT_PLAN.md)，环境与历史数据核查见[实验说明](../../ext_scripts/docs/A6000_EXPERIMENT_NOTES.md)。
- GPU 编号仅为示例，启动前必须重新确认物理卡空闲；正式配置分配后保持卡号不变，不干预其他用户进程。

### 端到端实测值如何得到

实测值来自在 GPU 上执行完整模型后的 CUDA Event 计时，**不是预测值，也不是单层算子耗时乘层数得到的端到端估计**。

1. 按原始模型配置，通过 Hugging Face `from_config()` 创建随机初始化的完整模型；输入为指定 batch size、sequence length 的全 1 token 张量。
2. 推理使用 `eval()` 和 `torch.no_grad()`，在 `model(*inputs)` 前后记录 CUDA Event；执行 `torch.cuda.synchronize()` 后读取毫秒数。
3. 训练使用 `train()`，分别计时模型前向和 `loss.backward()`。交叉熵损失的前向计算不在计时区间内，也没有优化器更新步骤；反向计时仍包含损失的反向传播。因此该耗时不是完整训练 step 的耗时。
4. 每轮采集 25 次，前向、反向分别排序，各取最快 5 次求平均；训练单轮结果为这两个平均值之和，推理单轮结果仅为前向平均值。
5. 每项配置在同一物理卡上独立运行 3 轮，报告以三轮端到端结果的中位数作为最终实测值。

```text
单轮推理耗时 = 前向最快 5 次的平均值
单轮训练耗时 = 前向最快 5 次的平均值 + 反向最快 5 次的平均值
最终实测值   = 3 轮正式测量耗时的中位数
```

训练不是先对每次前向与反向求和再选最快 5 次；前向、反向和端到端的三轮中位数也分别计算，前后向中位数之和不一定等于端到端中位数。OOM、失败、重复轮次不完整和实测值为零的样本不计入对应误差统计，不以零补齐。

计时实现沿用 [trace.py](../../neusight/Tracing/trace.py) 中的 `measure_e2e()`，采样参数为 `active=25`、`sample=5`；三轮汇总见 [summarize_a6000.py](../../ext_scripts/scripts/summarize_a6000.py)。原实现没有单独的固定端到端预热阶段，每次采样前的 `gc.collect()` 与 `torch.cuda.empty_cache()` 均在计时区间之外。这是“最快样本平均”的基准口径，不是所有采样的平均延迟，也不代表自回归解码或线上服务延迟。

### 算子类别实测与端到端实测的区别

算子分析通过 FX 计算图逐节点测量前向、反向与梯度累加耗时。除 Switch 使用完整图外，其他模型测量单层图，再按仓库的层标记扩展，汇总 Linear、BMM、VEC、Softmax 和 LayerNorm 等类别。算子反向总耗时按 `bw + acc` 计算。

这些算子汇总与完整模型端到端计时是独立测量，不能将类别耗时之和视为端到端实测，也不能把类别误差直接相加解释端到端误差。类别 MAPE 是工作负载级类别 APE 的平均值，不是所有单个算子实例 APE 的平均值。

### 本地报告与数据位置

以下路径均相对于仓库根目录，实验产物不提交 Git，不随仓库克隆分发：

| 内容 | 本地路径 |
|---|---|
| 中文结果报告 | [scripts/asplos/summary_a6000/REPORT_ZH.md](summary_a6000/REPORT_ZH.md) |
| 英文结果报告 | [scripts/asplos/summary_a6000/REPORT.md](summary_a6000/REPORT.md) |
| 各卡各轮测量记录与算子表 | `scripts/asplos/label_a6000/gpu*/run_*/NVIDIA_RTX_A6000/`，分别为 JSON 和 CSV |
| 预测结果与算子图 | `scripts/asplos/results_a6000/` |
| 端到端汇总与逐轮结果 | `scripts/asplos/summary_a6000/e2e.csv`、`scripts/asplos/summary_a6000/repeats.csv` |
| 算子类别统计 | `scripts/asplos/summary_a6000/operator_medians.csv`、`scripts/asplos/summary_a6000/operator_metrics.csv` |
| 失败、重试历史与验收结果 | `scripts/asplos/summary_a6000/failures.csv`、`scripts/asplos/summary_a6000/attempt_history.csv`、`scripts/asplos/summary_a6000/verification.json` |
| 环境、哈希、逐次状态、日志与 GPU 遥测 | `scripts/asplos/logs_a6000/` |

每轮 JSON 保存 `fw_latency`、`bwall_latency`、`e2e_latency`，单位为毫秒；它们是该轮筛选汇总后的结果，不包含全部 25 次的逐次计时值。试跑数据另存于 `label_a6000/smoke/`，不混入正式三轮统计。`summary` 阶段生成英文 `REPORT.md`；`REPORT_ZH.md` 是另行整理的本地中文版，重新汇总后需同步核对和更新。

## Steps

1. **(Optional) Run scripts to collect datasets from scratch:**
   1. Run `run_collect.py` on each target GPU to collect datasets.
   2. After collecting datasets, run `run_dataset.py` to process the collected datasets.
   3. Run the following scripts to collect measured latency of ML model executions:
      - `run_label_amd.py`
      - `run_label_nvidia.py`
      - `run_label_fusion.py`
      - `run_label_distributed.py`
   4. For distributed training, run `install_megatron.sh` beforehand to install Megatron-LM.

2. **(Optional) Train NeuSight with the collected dataset:**
   1. Run `run_train.py` to train NeuSight with the processed datasets.
   2. Additionally, run the following scripts to train NeuSight and baseline models with the processed datasets:
      - `run_train_amd_neusight.py`
      - `run_train_nvidia_habitat.py`
      - `run_train_nvidia_micro.py`
      - `run_train_nvidia_neusight.py`

3. **Use the provided evaluation scripts to predict model execution latencies:**
   - `run_pred_amd_neusight.py`
   - `run_pred_nvidia_habitat.py`
   - `run_pred_nvidia_micro.py`
   - `run_pred_nvidia_neusight.py`
   - `run_pred_nvidia_roofline.py`

4. **Compare predictions against ground truth:**
   - Use the provided `summarize.py` and `table.py` scripts. These will generate the summary of the prediction results and the table of the prediction results in the `summary` directory.
