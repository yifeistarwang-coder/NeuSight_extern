# A6000 实验运行入口

方案及口径以 `A6000_EXPERIMENT_PLAN.md`、`A6000_EXPERIMENT_NOTES.md` 为准。以下命令从仓库根目录执行；所有实验产物仅保留在本地，不提交 Git。

## 环境

```bash
virtualenv --python /usr/bin/python3 .venv_a6000
.venv_a6000/bin/python -m pip install -r ext_scripts/requirements-a6000.txt
```

固定 PyTorch 2.1.0、Transformers 4.38.1 和 NumPy 1.26.4；实际 CUDA/cuDNN 版本由环境探针记录。不要修改共享 Python 环境。

## 分阶段执行

每次运行前使用 `nvidia-smi` 重新确认两张物理卡空闲。示例中的 GPU 0、2 不代表永久可用；只有初次选择时可以替换编号，已有正式配置分配后不要改变卡号。

```bash
.venv_a6000/bin/python ext_scripts/scripts/run_a6000.py prepare --gpus 0 2
.venv_a6000/bin/python ext_scripts/scripts/run_a6000.py predict --gpus 0 2
.venv_a6000/bin/python ext_scripts/scripts/run_a6000.py smoke --gpus 0 2
.venv_a6000/bin/python ext_scripts/scripts/run_a6000.py summary --gpus 0 2
```

检查 `scripts/asplos/summary_a6000/smoke.csv`、`gpu_comparison.csv` 及原始算子图／标签对应情况，记录试跑异常与实际耗时后，进入正式运行：

```bash
.venv_a6000/bin/python ext_scripts/scripts/run_a6000.py formal --gpus 0 2
.venv_a6000/bin/python ext_scripts/scripts/run_a6000.py summary --gpus 0 2
.venv_a6000/bin/python ext_scripts/scripts/verify_a6000.py
```

- `prepare` 保存 24 项矩阵、GPU 环境与拓扑、依赖版本和冻结输入 SHA-256。
- `predict` 生成全部 24 项预测和算子图；调试时可用 `--names 配置名 ...` 限定预测，不改变正式矩阵。
- `smoke` 在两卡分别测量四个规定的小 batch 配置，并额外测量 GPT-3 2.7B 和 Switch 的推理配置。
- `formal` 要求八次基础试跑完整、四组双卡比较存在，然后按试跑耗时进行每卡 12 项配置分配，每项在同一卡独立采集三轮。
- 每卡只启动一个任务；启动前等待该卡无计算进程、显存占用小于 256 MiB 且利用率为零。超过五分钟仍被占用则停止启动新任务，不终止其他用户进程。通过 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`，进程内只见 `cuda:0`。
- 所有子进程固定随机种子 42、FP32/eager，关闭 matmul 和 cuDNN TF32；计时主体仍直接调用原有 `trace_graph()`。
- 如果系统只提供 64 位 `libcuda.so.1` 而缺少 PyTorch 2.1 使用的 `libcuda.so` 名称，worker 在独立环境的 `driver_libs/` 建立指向同一已安装驱动的别名并重新启动自身；不修改系统目录或驱动版本。实际动态库搜索路径写入每次环境记录。
- 每次尝试记录原始输出哈希、错误类别、运行耗时、代码哈希、GPU 时钟／温度／功耗及计算进程采样；旧的有效结果在验证后跳过，不覆盖。
- 冻结输入不一致、没有状态文件的异常退出及已存在但未验证的输出会停止执行，需先人工检查。正常 OOM／trace／parse 失败写入状态并继续其他配置；再次运行可重试失败任务。
- 定向重试可用 `formal --gpus 0 2 --names 配置名 ...`；保持原分配和轮次，成功结果不覆盖，重试前将失败状态完整保存到该轮 `history/` 子目录，原始日志和采样文件也保留。

## 汇总与验收

`scripts/asplos/summary_a6000/` 保存端到端及每轮误差、训练前后向分解、五类算子误差、测量波动、失败清单、双卡试跑比较和共同有效配置的旧 GPU 对照。`REPORT.md` 是本地结果说明，`audit.json` 是产物完整性摘要。

主指标只使用完整三轮的中位数；部分结果不伪装成完整重复。零值、缺失和失败不计零误差。原始算子反向统一按 `bw + acc` 汇总；旧结果的修正分解另存本地，不覆盖仓库历史数据。

`audit.json` 中的终止次数与成功次数必须区分：OOM 等明确失败可以计为已执行，但不能计为成功测量。最终还需核对日志、输入哈希、节点对应、进程采样及文档交付标准，不能仅凭汇总脚本正常退出判定完成。
