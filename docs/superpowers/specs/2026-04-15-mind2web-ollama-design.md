# Mind2Web 实验 Ollama 复现设计文档

## 1. 背景

当前仓库的 Mind2Web 在线实验主路径依赖 OpenAI：

- [`run_mind2web.py`](/home/wzh/Synapse/run_mind2web.py) 调用 [`synapse/agents/mind2web.py`](/home/wzh/Synapse/synapse/agents/mind2web.py) 执行逐步动作预测
- [`synapse/utils/llm.py`](/home/wzh/Synapse/synapse/utils/llm.py) 使用 OpenAI Chat API
- [`synapse/memory/mind2web/build_memory.py`](/home/wzh/Synapse/synapse/memory/mind2web/build_memory.py) 使用 `OpenAIEmbeddings` 构建 FAISS memory

本设计的目标是在不复现微调实验的前提下，将 Mind2Web 在线实验部分改造为基于 Ollama 的本地复现方案，并统一使用 Ollama 提供的 OpenAI-compatible 接口。

## 2. 目标与非目标

### 2.1 目标

1. 使用 Ollama 的 OpenAI-compatible chat 接口替换当前 OpenAI Chat API。
2. 使用 Ollama 的 OpenAI-compatible embeddings 接口替换当前 `OpenAIEmbeddings`。
3. 支持至少两种 chat 模型：
   - `qwen3.5:4b`
   - `qwen3.5:9b`
4. 默认使用 embedding 模型：
   - `qwen3-embedding:0.6b`
5. 保持 Mind2Web 核心实验逻辑不变，包括：
   - exemplar memory 检索流程
   - trajectory prompt 组织方式
   - 候选元素过滤逻辑
   - 评测指标与口径
6. 后续能够通过参数切换到其他 Ollama 模型，而不改动 agent 主逻辑。

### 2.2 非目标

1. 不复现微调实验。
2. 不改造 [`build_dataset.py`](/home/wzh/Synapse/build_dataset.py)。
3. 不改造 [`finetune_mind2web.py`](/home/wzh/Synapse/finetune_mind2web.py)。
4. 不改造 [`evaluate_mind2web.py`](/home/wzh/Synapse/evaluate_mind2web.py)。
5. 不保留旧 OpenAI 分支兼容。
6. 不新增独立的 `run_mind2web_ollama.py` 或 `build_memory_ollama.py`。
7. 不修改 prompt 策略、memory 检索策略、评测指标定义或上下文裁剪顺序。

## 3. 方案选型

本次在三种候选方案中选择“**Ollama-only 但分层重构**”：

### 3.1 候选方案

1. 最小改动直替
   直接把现有 OpenAI 调用替换成 Ollama 接口，其他结构尽量不动。
2. Ollama-only 但分层重构
   保持实验入口与评测流程不变，将 chat 与 embedding 的模型访问逻辑下沉到独立适配层。
3. 新增独立 Ollama 实验分支
   额外复制一套 `run/build_memory` 代码专门服务 Ollama。

### 3.2 选择结果

选择方案 2，原因如下：

1. 仍然是 Ollama-only，满足当前简化目标。
2. 比方案 1 更易维护，避免把 OpenAI 假设硬编码在 Ollama 流程中。
3. 比方案 3 更少重复代码，后续切换其他 Ollama 模型成本更低。

## 4. 设计原则

1. **实验逻辑不动**
   尽量保持原始 Mind2Web 在线实验的数据流、prompt 组织、memory 检索和评测行为。
2. **模型依赖下沉**
   Provider 差异仅存在于 utils 层，不扩散到 agent 和 env 逻辑。
3. **参数驱动**
   通过 CLI 参数指定 Ollama 服务地址、chat 模型与 embedding 模型。
4. **结果可复现**
   结果目录、memory 目录和日志中显式记录模型信息。
5. **与源代码行为一致**
   上下文长度控制与超限处理保持源代码思路，不重新设计裁剪顺序。

## 5. 当前代码现状分析

### 5.1 在线实验入口

[`run_mind2web.py`](/home/wzh/Synapse/run_mind2web.py) 负责：

1. 读取 benchmark 数据。
2. 从 `scores_all_data.pkl` 中为候选元素补齐 `score/rank`。
3. 调用 [`synapse/agents/mind2web.py`](/home/wzh/Synapse/synapse/agents/mind2web.py) 中的 `eval_sample()` 执行逐样本评测。

### 5.2 Chat 调用

[`synapse/utils/llm.py`](/home/wzh/Synapse/synapse/utils/llm.py) 当前特点：

1. 直接初始化 OpenAI client。
2. `generate_response()` 走 OpenAI chat/completion API。
3. `num_tokens_from_messages()` 和 `MAX_TOKENS` 依赖 OpenAI 模型名与 `tiktoken`。

这些逻辑与 Ollama 不兼容，是在线实验改造的核心点。

### 5.3 Memory 构建

[`synapse/memory/mind2web/build_memory.py`](/home/wzh/Synapse/synapse/memory/mind2web/build_memory.py) 当前特点：

1. 从 train 数据构造 `specifier` 和 exemplar 对话。
2. 使用 `OpenAIEmbeddings(model="text-embedding-ada-002")` 构建 FAISS。
3. 输出 `index.faiss`、`index.pkl` 与 `exemplars.json`。

这里需要替换 embedding 来源，但不改变 exemplar 构造与 FAISS 使用方式。

## 6. 总体架构

整体流程保持为：

1. 使用 [`build_memory.py`](/home/wzh/Synapse/build_memory.py) 构建 Mind2Web memory。
2. 使用 [`run_mind2web.py`](/home/wzh/Synapse/run_mind2web.py) 运行 `test_task`、`test_website`、`test_domain`。
3. 将逐样本日志写入 `results/mind2web/...`。
4. 继续使用现有指标：
   - `element_acc`
   - `action_f1`
   - `step_success`
   - `success`

系统划分为四层。

### 6.1 实验入口层

文件：

- [`build_memory.py`](/home/wzh/Synapse/build_memory.py)
- [`run_mind2web.py`](/home/wzh/Synapse/run_mind2web.py)

职责：

1. 解析 CLI 参数。
2. 组装数据目录、memory 路径和日志路径。
3. 将模型与服务配置传给下层。
4. 不承担具体 API 调用细节。

### 6.2 Agent 推理层

文件：

- [`synapse/agents/mind2web.py`](/home/wzh/Synapse/synapse/agents/mind2web.py)

职责：

1. 保持当前 trajectory prompt 组织逻辑。
2. 保持 exemplar 注入逻辑。
3. 保持动作解析和指标计算流程。
4. 不直接依赖 OpenAI SDK，而是调用统一 chat 访问接口。

### 6.3 模型访问层

文件：

- [`synapse/utils/llm.py`](/home/wzh/Synapse/synapse/utils/llm.py)
- 新增 `synapse/utils/embeddings.py`

职责：

1. `llm.py` 负责 chat 请求发送、响应抽取、上下文预算判断与基础错误包装。
2. `embeddings.py` 负责调用 Ollama OpenAI-compatible embeddings 接口并返回向量。
3. 业务层只感知“聊天补全”和“文本向量化”两个能力，不感知具体 provider 细节。

### 6.4 Memory 检索层

文件：

- [`synapse/memory/mind2web/build_memory.py`](/home/wzh/Synapse/synapse/memory/mind2web/build_memory.py)

职责：

1. 保留 `specifier` 生成逻辑。
2. 保留 exemplar 构建逻辑。
3. 保留 FAISS 作为向量索引。
4. 仅将 embedding 来源替换为 Ollama。

## 7. 配置设计

本次只保留最小必要配置。

### 7.1 服务连接配置

1. `--api_base`
   - 默认值：`http://localhost:11434/v1`
   - 用途：指定 Ollama OpenAI-compatible API 根路径。
2. `--api_key`
   - 默认行为：优先读取环境变量；若为空，则回退为固定占位值 `ollama`
   - 用途：兼容 OpenAI-style client 初始化要求

### 7.2 实验模型配置

1. `--chat_model`
   - 示例：`qwen3.5:4b`、`qwen3.5:9b`
2. `--embedding_model`
   - 默认值：`qwen3-embedding:0.6b`
3. `--max_context_tokens`
   - 用途：为 Ollama 模型提供显式上下文预算

### 7.3 保持不变的现有实验参数

以下 Mind2Web 参数继续保留：

1. `--benchmark`
2. `--previous_top_k_elements`
3. `--top_k_elements`
4. `--retrieve_top_k`
5. `--no_memory`
6. `--no_trajectory`
7. `--start_idx`
8. `--end_idx`

## 8. 数据流设计

### 8.1 Memory 构建链路

1. [`build_memory.py`](/home/wzh/Synapse/build_memory.py) 接收 `mind2web_data_dir`、`api_base`、`api_key`、`embedding_model` 等参数。
2. [`synapse/memory/mind2web/build_memory.py`](/home/wzh/Synapse/synapse/memory/mind2web/build_memory.py) 读取 train 数据与 `scores_all_data.pkl`。
3. 为每个样本构造：
   - `specifier`
   - exemplar 对话
4. 调用 `synapse/utils/embeddings.py` 获取每个 `specifier` 的向量。
5. 使用 FAISS 建索引并落盘。
6. 输出：
   - `index.faiss`
   - `index.pkl`
   - `exemplars.json`
   - `memory_meta.json`

### 8.2 `memory_meta.json` 设计

建议记录以下字段：

1. `embedding_model`
2. `api_base`
3. `top_k_elements`
4. `num_exemplars`
5. `created_at`

作用：

1. 便于定位 memory 使用的 embedding 配置。
2. 在运行实验时做最小一致性校验，避免错误复用不匹配的 memory。

### 8.3 在线实验链路

1. [`run_mind2web.py`](/home/wzh/Synapse/run_mind2web.py) 读取 benchmark 数据。
2. 使用 `scores_all_data.pkl` 给候选元素补充 `score/rank`。
3. [`synapse/agents/mind2web.py`](/home/wzh/Synapse/synapse/agents/mind2web.py)：
   - 构造当前 sample 的 specifier
   - 从 FAISS 中检索 exemplar
   - 拼接 trajectory prompt
4. 通过 `synapse/utils/llm.py` 调用 Ollama `/v1/chat/completions`。
5. 解析输出动作。
6. 计算并写入现有指标及会话日志。

## 9. 路径与产物设计

为避免不同模型结果互相覆盖，目录按模型隔离。

### 9.1 Memory 目录

建议路径：

`synapse/memory/mind2web/<embedding_model_slug>/top<k>/`

示例：

`synapse/memory/mind2web/qwen3-embedding-0.6b/top3/`

### 9.2 Results 目录

建议路径：

`results/mind2web/<chat_model_slug>/<benchmark>/<mode>/`

示例：

1. `results/mind2web/qwen3.5-4b/test_domain/with_memory/`
2. `results/mind2web/qwen3.5-9b/test_domain/no_mem/`
3. `results/mind2web/qwen3.5-4b/test_task/no_mem_no_traj/`

这样可以：

1. 清晰区分不同 chat 模型的结果。
2. 清晰区分是否使用 memory / trajectory。
3. 支持后续扩展到其他 Ollama 模型。

## 10. 错误处理与稳定性设计

### 10.1 Memory 构建失败策略

`build_memory` 采用 fail-fast：

1. embedding 请求失败时立即报错。
2. 不允许静默跳过部分样本。
3. 不允许生成不完整的 memory 索引。

原因是 memory 一旦构建不完整，后续所有实验结果都不可信。

### 10.2 在线实验失败策略

`run_mind2web` 采用 step-level 失败记录：

1. 某一步 chat 请求失败时，不中断整批 benchmark。
2. 该 step 记为失败，并在该样本日志中记录错误信息。
3. 继续执行后续样本，保证整批实验可跑完并保留失败证据。

### 10.3 输出规范化

Ollama 上的小模型可能出现以下输出偏差：

1. 先解释原因再给 action。
2. 不带反引号。
3. 尾部带 `</s>` 或多余换行。
4. 输出多个候选动作。

因此在 chat 层增加统一规范化逻辑：

1. 优先提取反引号中的动作。
2. 若没有反引号，再用正则匹配首个合法 `CLICK|TYPE|SELECT [...]`。
3. 清理尾部结束标记与多余空白。
4. 若仍无法解析，记为非法动作。

该规范化只做输出清洗，不改变指标计算逻辑。

### 10.4 上下文控制

本设计不改变源代码的上下文裁剪顺序和整体行为，只做适配性修改：

1. 将当前 OpenAI/tiktoken 强绑定的长度判断改为 Ollama 可用的实现。
2. 保持原有 exemplar 使用逻辑与 trajectory 拼接方式。
3. 保持原有超限处理思路，不额外引入新的裁剪策略。

这样可以最大程度保证复现的是原实验逻辑，而不是新的 prompt 管理策略。

## 11. 评测一致性设计

评测口径保持不变。

### 11.1 保持不变的评测基础

1. 候选元素仍来自 `scores_all_data.pkl`。
2. HTML 裁剪仍由 [`synapse/envs/mind2web/env_utils.py`](/home/wzh/Synapse/synapse/envs/mind2web/env_utils.py) 中现有工具函数完成。
3. action 解析继续与现有 `parse_act_str()` 兼容。
4. 文本 F1 计算继续使用现有 `calculate_f1()`。

### 11.2 保持不变的指标

继续统计以下指标：

1. `element_acc`
2. `action_f1`
3. `step_success`
4. `success`

### 11.3 评测改造边界

只替换模型访问方式与日志补充字段，不改变：

1. ground-truth 对齐逻辑
2. action 格式定义
3. sample 级别和 step 级别打分方式

## 12. 日志设计

保留当前逐样本 JSON 日志结构，并增加复现实验所需字段。

建议新增字段：

1. `chat_model`
2. `embedding_model`
3. `api_base`
4. `raw_response`
5. `normalized_action`
6. `error_type`

作用：

1. 排查模型输出格式问题。
2. 排查 embedding 与 memory 是否匹配。
3. 支持 `qwen3.5:4b` 与 `qwen3.5:9b` 的横向对比。

## 13. 实施范围

本次实施只包含以下改造。

### 13.1 需要改造的文件

1. [`synapse/utils/llm.py`](/home/wzh/Synapse/synapse/utils/llm.py)
   - 替换为 Ollama OpenAI-compatible chat 调用
   - 适配新的上下文预算判断
2. 新增 `synapse/utils/embeddings.py`
   - 提供 Ollama embeddings 访问封装
3. [`synapse/memory/mind2web/build_memory.py`](/home/wzh/Synapse/synapse/memory/mind2web/build_memory.py)
   - 替换 `OpenAIEmbeddings`
   - 增加 `memory_meta.json`
4. [`build_memory.py`](/home/wzh/Synapse/build_memory.py)
   - 增加服务与 embedding 相关参数
5. [`run_mind2web.py`](/home/wzh/Synapse/run_mind2web.py)
   - 增加服务与 chat 相关参数
6. [`synapse/agents/mind2web.py`](/home/wzh/Synapse/synapse/agents/mind2web.py)
   - 接入新 chat 层
   - 增加输出规范化与日志补充字段

### 13.2 明确不改造的文件

1. [`build_dataset.py`](/home/wzh/Synapse/build_dataset.py)
2. [`finetune_mind2web.py`](/home/wzh/Synapse/finetune_mind2web.py)
3. [`evaluate_mind2web.py`](/home/wzh/Synapse/evaluate_mind2web.py)

## 14. 运行方式

建议将文档中的运行流程分成三步。

### 14.1 启动 Ollama 并准备模型

需要准备：

1. chat 模型：
   - `qwen3.5:4b`
   - `qwen3.5:9b`
2. embedding 模型：
   - `qwen3-embedding:0.6b`

### 14.2 构建 memory

按 embedding 模型构建一次 memory，并确保其产物目录中包含 `memory_meta.json`。

### 14.3 跑三个 benchmark

分别执行：

1. `test_task`
2. `test_website`
3. `test_domain`

并分别对：

1. `qwen3.5:4b`
2. `qwen3.5:9b`

产出独立结果目录。

## 15. 结果产出与汇总

### 15.1 Memory 产物

1. `index.faiss`
2. `index.pkl`
3. `exemplars.json`
4. `memory_meta.json`

### 15.2 逐样本日志

保持现有逐样本 JSON 日志，并补充模型与错误信息字段。

### 15.3 汇总结果

本轮设计先定义汇总口径，不强制新增汇总脚本。

建议至少汇总：

1. `element_acc` 均值
2. `action_f1` 均值
3. `step_success` 均值
4. `episode success rate`

## 16. 测试与验收标准

### 16.1 实施后最小验证

1. 能在不依赖 OpenAI API 的情况下完成 Mind2Web memory 构建。
2. 能在 Ollama 本地服务上跑通 [`run_mind2web.py`](/home/wzh/Synapse/run_mind2web.py)。
3. 能分别为 `qwen3.5:4b` 和 `qwen3.5:9b` 产出独立结果目录。
4. 日志中能看到原始输出、规范化动作和错误信息。

### 16.2 验收标准

1. 在线实验只依赖 Ollama OpenAI-compatible API。
2. memory 检索与 chat 推理均不再依赖 OpenAI 官方服务。
3. 评测口径与源代码保持一致。
4. 上下文裁剪顺序与源代码保持一致。
5. 后续替换其他 Ollama 模型时，无需改动 agent 主逻辑。

## 17. 风险与后续关注点

### 17.1 已知风险

1. 小模型可能更容易输出额外解释性文本，导致动作解析失败率高于原模型。
2. 不同 Ollama 模型的上下文上限和输出风格差异可能影响复现稳定性。
3. embedding 模型替换后，memory 检索分布可能与原始 OpenAI embedding 不同。

### 17.2 后续可选扩展

1. 增加轻量汇总脚本，自动汇总 benchmark 指标。
2. 增加 memory 配置一致性检查，在运行前比对 `memory_meta.json`。
3. 在完成本轮在线实验复现后，再单独评估是否需要扩展到微调链路。

## 18. 成功标准总结

本项目在本轮设计中的成功标准是：

1. 以 Ollama-only 方式复现 Mind2Web 在线实验。
2. 使用 `qwen3.5:4b`、`qwen3.5:9b` 与 `qwen3-embedding:0.6b` 跑通主流程。
3. 保持原有 prompt、检索、评测和上下文裁剪顺序不变。
4. 让后续切换其他 Ollama 模型仅需调整参数，而不必重写实验逻辑。
