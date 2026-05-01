# Llama Matthew Perturbation Evaluation Status

## 1. 当前实验进度 (Current Progress)
我们成功在 Llama-3.2-1B 与 3B Instruct 模型上实施了“能量守恒马太微扰”（Matthew Operator）实验。实验通过对 MLP 层的 `up_proj` 和 `down_proj` 进行奇异值分解（SVD），并在对数空间下对奇异值谱进行缩放（$\alpha$），从而观察模型的人格特征偏移。

- **完成模型**：
  - Llama-3.2-1B-Instruct
  - Llama-3.2-3B-Instruct
- **覆盖参数**：$\alpha \in \{-0.2, -0.1, 0.0, 0.1, 0.2\}$
- **评测维度**：
  1. **标准认知 (Standard)**：涵盖 GSM8K, MATH, ARC-Challenge, DROP（No-CoT）。
  2. **独立 CoT (Standard-CoT)**：仅保留 GSM8K 与 MATH，用于观察 thinking/CoT 行为与格式稳定性。
  3. **工具调用 (BFCL)**：AST 级语法与语义验证，包含 3 次重试机制。
  4. **人格法庭 (Judge)**：基于 TruthfulQA, HaluEval, AdvBench 的 LLM-as-a-judge 评估；默认 judge model 为 `v4-flash`，调用时解析为 `deepseek-v4-flash`。
  5. **扩展基准 (lm-eval)**：按语义分组覆盖 Knowledge/MMLU、Hard Science/GPQA、Exam Reasoning/AGIEval 英文子项、Commonsense/HellaSwag、Instruction Following/IFEval。

## 2. 已发现的核心结论 (Key Findings)
- **左脑/右脑分化趋势**：
  - **$\alpha > 0$ (Sharpening/尖峰化)**：显著提升逻辑严密性与数学性能，但在超轻量模型（1B）上会较早导致指令遵循（Format Compliance）的崩塌。
  - **$\alpha < 0$ (Smoothing/平滑化)**：提升了回复的创意联想与对齐稳定性，但会导致多步推理逻辑变弱及幻觉率上升。
- **Bad Case 捕获**：实验成功记录了 TruthfulQA 中的典型幻觉案例，为后续人格精调提供了数据支撑。

## 3. 已踩坑与技术挑战 (Status & Gotchas)
- **[已修复] vLLM KV Cache OOM**：3B 模型由于本身权重占用较大，在 vLLM 默认 128k 上下文配置下会导致显存溢出。
  - *解决方案*：在评测脚本中显式将 `max_model_len` 限制在 4096，并将 `gpu_memory_utilization` 设为 0.6。
- **[已修复] HuggingFace 权限冲突**：在共享 HPC 环境下，Gated Repo（如 Llama-3.2）的鉴权与 `.locks` 目录的读写权限会导致脚本崩溃。
  - *解决方案*：使用本地 Snapshot 绝对路径加载权重，并重定向 `HF_HOME` 到私有目录。
- **[已修复] 结果文件名冲突**：初步脚本未在结果文件名中区分模型 ID，导致 1B 与 3B 结果互相覆盖。
  - *解决方案*：重构 `out_path` 逻辑，文件名现在包含模型名称及 Alpha 参数。
- **[已修复] 结果目录不一致**：脚本默认写入路径与仓库内已提交结果目录不一致。
  - *解决方案*：脚本默认输出目录改为 `results/standard`、`results/bfcl`、`results/judge` 下的现有结构，并暴露 `--output_root`。
- **[已改进] 多卡利用不足**：原脚本固定 `tensor_parallel_size=1`，即使 Slurm 分配多张卡也只用一张。
  - *解决方案*：vLLM 脚本新增 `--tensor_parallel_size`、`--gpu_memory_utilization`、`--max_model_len` 参数。
- **[已改进] 临时目录冲突**：原临时目录只按 alpha 命名，并发跑不同模型或重复任务时可能互相覆盖。
  - *解决方案*：改用 `tempfile.mkdtemp` 创建带模型名和 alpha 前缀的唯一目录。
- **[已验证] Qwen3-8B 远端烟测**：`wzq` 不能直连 Hugging Face，但 `HF_ENDPOINT=https://hf-mirror.com` 可用。
  - *验证结果*：在 `ailab01` 的 `rtx5880ada` 上用 Slurm、uv Python 3.11、vLLM eager 模式跑通 `Qwen/Qwen3-8B` + HellaSwag 5 样本，结果 `acc=0.6`、`acc_norm=0.8`。
  - *注意事项*：系统 Python 3.12 缺少 `Python.h`，会导致 Triton gcc 编译失败；需使用 `uv python install 3.11` 后重建 `.venv`。混卡节点建议设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。

## 4. 下一步计划 (Next Steps)
- 汇总 1B 与 3B 的全量数据，绘制五维人格雷达图对比。
- 探索马太算子在更大规模模型（如 8B, 70B）上的可迁移性。
- 定量分析“格式崩塌点”与模型参数量及 $\alpha$ 值的数学关系。

## 5. 最新 3B 模型追加结论 (Latest 3B Findings)
- **推理爆发 (Reasoning Surge)**：在 `alpha=0.2` (Sharpening) 作用下，Llama-3.2-3B 的 GSM8K (CoT) 准确率从 **0.40** 跃升至 **0.57** (+17% 绝对增益)。
- **格式增益 (Format Synergy)**：尖峰化算子在提升 3B 模型逻辑的同时，意外地将格式错误率从 **0.50** 降低到了 **0.20**。
- **事实代价 (Factual Trade-off)**：对应的 TruthfulQA 得分略有下降（1.13 -> 1.07），证明了马太微扰在左脑强化时对右脑事实记忆的轻微挤压效应。

## 6. Result summaries and archives

- Llama-3.2-1B alpha sweep summary: `docs/llama1b_alpha_sweep_results.md`
- Qwen3-8B current result status: `docs/qwen3_8b_result_status.md`
- Raw archive manifest: `docs/result_archives.md`
