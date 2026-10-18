# agentdoc —— 智能文档问答 Agent（方案复现工程）

[![CI](https://github.com/cheneryang0616-bot/agentdoc-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/cheneryang0616-bot/agentdoc-assistant/actions/workflows/ci.yml)

> 一个可**零网络、零 API Key** 跑通，也**可一键接入真实嵌入模型与真实 LLM** 的工程项目，
> 复现《Agent开发项目》方案中的 **AgentLoop / 多级记忆 / 三级上下文压缩 / SubAgent 记忆整理 / RAG 微调**。
> 环境：Python 3.9+、numpy；真实模型为可选增强。

---

## 1. 与方案文档的对应关系

| 方案文档中的要求 | 本工程实现 | 位置 |
|---|---|---|
| AgentLoop（解决多跳问题） | `LoopState` / `run_one_turn` / `agent_loop` | `agentdoc/loop.py` |
| Tool 定义（RAG Search / Memory / SubAgent） | 工具注册表 + 5 个工具 | `agentdoc/tools/` |
| 三级上下文压缩（L1 落盘 / L2 占位符 / L3 全量） | `ContextManager.compress` | `agentdoc/context/manager.py` |
| SubAgent 记忆整理（Agent Team） | `run_subagent` + `subagent` 工具 | `agentdoc/subagent.py` |
| CronJob / HEARTBEAT（Auto-Dream） | `memory_consolidate` 后台整理 | `agentdoc/memory/lifecycle.py` |
| 工作 / 情景 / 语义 三级记忆 | `MemoryStore` | `agentdoc/memory/store.py` |
| 时间衰减 + 重要性混合评分 | `time_decay` / `hybrid_score` | `agentdoc/memory/scoring.py` |
| 记忆生命周期（阈值提升/合并） | `consolidate` / `promote_*` / `merge_semantic` | `agentdoc/memory/lifecycle.py` |
| MarkItDown 规范化 + 语义分块 | `normalize_source` / `semantic_chunk` | `agentdoc/rag/ingest.py` |
| 384 维向量 + 维度容错（补齐/截断） | `fix_dim` / `HashingEmbedder` | `agentdoc/rag/embedder.py` |
| MQE + HyDE + 扩展-检索-合并 | `expand_queries` / `hyde_pseudo_doc` / `rrf_fuse` | `agentdoc/rag/retriever.py` |
| RAG 微调：数据蒸馏 + 三元损失 | `experiments/exp4_*` | numpy 手写反向传播 |

---

## 2. 后端矩阵（可插拔）

| 层 | 后端 | 触发方式 | 说明 |
|---|---|---|---|
| 嵌入 | `HashingEmbedder`（1024 维） | `AGENTDOC_EMBEDDER=hash` | 零依赖、确定性，离线兜底 |
| 嵌入 | `SentenceTransformerEmbedder`（BGE-512 维） | `AGENTDOC_EMBEDDER=st` 或 `auto` | 真实中文检索模型 |
| LLM | `OfflineBackend`（规则规划器） | `AGENTDOC_LLM_PROVIDER=offline` 或自动兜底 | 无 Key 也能完整演示 AgentLoop |
| LLM | `OpenAICompatBackend` | `AGENTDOC_LLM_PROVIDER=deepseek\|dashscope\|qwen\|moonshot\|openai\|ollama\|vllm` | 走标准库 HTTP，无需 SDK |
| LLM | `AnthropicBackend` | `AGENTDOC_LLM_PROVIDER=anthropic` | 需 `pip install anthropic` |

`auto` 策略：嵌入优先用**已缓存的** BGE 模型（不联网，约 3s 加载），否则回退哈希；
LLM 有任意 Key/BaseURL 就用真实模型，否则回退离线规划器。

---

## 3. 快速开始

```bash
pip install -r requirements.txt        # 仅 numpy；真实模型见第 4 节

python3 -m agentdoc.cli                # 交互式问答（q 退出）
python3 -m agentdoc.cli --demo         # 内置演示：多跳 / 记忆写读 / 记忆整理
bash run_all.sh                        # 一键：单测 + 6 个实验 + 后端自检 + Demo

python3 scripts/verify_backends.py     # 查看当前实际使用的嵌入/LLM 后端
python3 scripts/mcp_demo.py           # MCP 端到端演示（stdio）
python3 scripts/mcp_host_demo.py      # MCP Host：连接多个外部 server

# 用真实 LLM：先把 Key 写入 ~/.agentdoc.env（参考 .env.example）
cp .env.example ~/.agentdoc.env && vim ~/.agentdoc.env
bash scripts/run_with_llm.sh --demo    # 带真实模型跑演示
```

---

## 4. 接入真实模型

### 4.1 真实嵌入（BGE，中文）

```bash
# 1) 安装（国内建议走清华镜像）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers

# 2) 若 transformers 报 Keras3 错误，设置（本工程已自动设置）
export USE_TF=0

# 3) 拉模型（走 HF 镜像）
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-zh-v1.5')"

# 4) 启用
export AGENTDOC_EMBEDDER=st        # 或 auto（检测到本地缓存即启用）
```

### 4.2 真实 LLM（任选其一）

```bash
# DeepSeek
export AGENTDOC_LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-xxx
export AGENTDOC_MODEL=deepseek-chat

# 通义千问（DashScope OpenAI 兼容模式）
export AGENTDOC_LLM_PROVIDER=dashscope
export DASHSCOPE_API_KEY=sk-xxx
export AGENTDOC_MODEL=qwen-plus

# 本地 Ollama（无需 Key）
export AGENTDOC_LLM_PROVIDER=ollama
export AGENTDOC_MODEL=qwen2.5:7b

# 任意 OpenAI 兼容服务
export AGENTDOC_BASE_URL=https://your-host/v1
export AGENTDOC_API_KEY=sk-xxx
export AGENTDOC_MODEL=your-model
```

> 验证：`python3 scripts/verify_backends.py --smoke`

> **实测记录（2026-10）**：`deepseek-chat` 已跑通完整 Agent 流程——
> 多跳问题会一次性并行发起 2 次 `rag_search`；记忆写入/读取、SubAgent 记忆整理均正确触发。
> 注意：DeepSeek 的 Key 需要带 `sk-` 前缀，漏写会返回 401 `api key is invalid`。

---

## 5. 关键设计

### 5.1 AgentLoop（多跳）
模型产出 `tool_use` → 工程侧执行 → 结果回灌 → 继续，直到 `stop_reason != "tool_use"`。
多跳问题会被拆成多个子问题逐次检索。

### 5.2 三级上下文压缩（成本递增级联）
1. **L1**：>400 字符的工具结果落盘，正文换成占位符（可用 `file_read` 回读）；
2. **L2**：久远 QA 替换为占位符（保留最近 2 轮）；
3. **L3**：仍超预算则整段压缩为一条摘要。
压缩在**每次交互之前**执行。

### 5.3 记忆系统
工作记忆（内存）→ 重要性 ≥ 0.6 提升为情景记忆 → 访问 ≥ 2 次抽象为语义记忆 →
相似度 ≥ 0.45 自动合并。检索用混合评分：
`0.55·语义相关度 + 0.30·重要性 + 0.15·时间新鲜度`。

### 5.4 SubAgent 记忆整理
`subagent` 工具启动**独立上下文**的子 Agent，只做：读记忆 → `memory_consolidate`
（提升/合并/清理）→ 输出报告。对应 Auto-Dream。

### 5.5 RAG 微调（两阶段）
- **阶段 A**：LLM 蒸馏打标 + 人类对齐到 95% + 90/10 切分；
- **阶段 B**：`(query, pos_doc, neg_doc)` 三元组 + margin 损失，numpy **手写反向传播**
  训练线性投影（恒等初始化，从基线出发做增量修正）。

### 5.6 MCP 工具化（Model Context Protocol）

**双向支持**，纯标准库实现（JSON-RPC 2.0 over stdio，不依赖官方 SDK）：

| 方向 | 组件 | 作用 |
|---|---|---|
| **作 server** | `agentdoc/mcp/server.py` | 把自己的能力暴露给外部客户端 |
| **作 host** | `agentdoc/mcp/host.py` | 同时连接多个外部 server，聚合工具 |
| **单连** | `agentdoc/mcp/client.py` | 连接单个 server + 工具适配器 |

```bash
python3 scripts/mcp_demo.py          # 方向一：agentdoc 当 server
python3 scripts/mcp_host_demo.py     # 方向二：agentdoc 当 host，连 2 个外部 server
python3 -m agentdoc.mcp.server --verbose
```

协议握手流程：

```
initialize  ->  notifications/initialized  ->  tools/list  ->  tools/call
```

**作 server 时**暴露 `rag_search` / `memory_read` / `memory_write`（带 JSON Schema）。

**作 host 时**通过 `mcp_servers.json`（兼容 Claude Desktop 写法）管理多个 server，
工具按 `<server>__<tool>` 命名空间聚合，避免同名冲突：

```json
{
  "mcpServers": {
    "agentdoc": {"command": "python3", "args": ["-m", "agentdoc.mcp.server"]},
    "fs":       {"command": "python3", "args": ["examples/mcp_fs_server.py"]}
  }
}
```

实测：host 同时挂载 `agentdoc`（检索/记忆）与 `fs`（文件系统），聚合出 6 个工具；
把 `agentdoc__rag_search` 覆盖为本地的 `rag_search` 后，多跳问题仍触发 2 次 MCP 调用——
**AgentLoop / 工具注册表 / 记忆系统均无需修改**。

---

## 6. 实验结果

见 [`results/RESULTS.md`](results/RESULTS.md)。关键数字（真实运行产出）：

| 实验 | 指标 | 结果 |
|---|---|---|
| 1 · AgentLoop | 哈希检索器：多跳覆盖率 | 0.875 → **1.000** |
| 1 · AgentLoop | BGE 检索器：多跳覆盖率 | 1.000 → 1.000（强检索器下单次已足够）|
| 2 · 上下文压缩 | L1 场景 | 19218 → 1242 tokens（0.065） |
| 2 · 上下文压缩 | L2 / L3 场景 | 均触发 |
| 3 · 记忆 | 工作→情景 / 情景→语义 / 合并 | 3 / 1 / 1 |
| 4 · 微调 | 全量 Recall@1 | 0.828 → **0.969** |
| 4 · 微调 | 5 折 CV Recall@1 | 0.830 → **0.862** |
| 5 · 嵌入对比 | 哈希 → BGE，Recall@1 (dense) | 0.828 → **0.922** |
| 5 · 嵌入对比 | 哈希 → BGE，Recall@1 (MQE) | 0.891 → **0.938** |

> **一个值得讲的发现**：AgentLoop 的收益**依赖检索器强弱**。用弱检索器（哈希）时，
> 单次检索只能覆盖 0.875，逐子问题检索能到 1.000；换成强检索器（BGE）后，
> 单次 top-5 已经能覆盖全部子问题，Loop 的边际收益归零。这说明——
> **“该不该上 Agent 架构”要先看检索质量，不要为了 Agent 而 Agent。**

### 6.1 失败分析（16 个对抗用例）

详见 [`results/ERROR_ANALYSIS.md`](results/ERROR_ANALYSIS.md)。

| 配置 | 通过 | 主要失败 |
|---|---|---|
| offline（规则规划器） | 13/16 | 完全不会拒答，库外问题硬拼无关文档 |
| offline + 相关性闸门 | **15/16** | 只剩“近似但无答案” |
| deepseek-chat | 14/16 | 能识别库外，但越界用通用知识作答 |
| deepseek + 闸门 + 约束提示 | **15/16** | 只剩“近似但无答案” |

> 结论：**所有失败都是“越界”而不是“召回不足”**——小知识库上检索不是瓶颈，
> 边界判断才是。相关性阈值能修掉“明显库外”，但修不掉“语义相近但没答案”，
> 后者需要答案级 groundedness 校验。

---

## 7. 诚实声明（复现边界）

1. **默认是无 Key 可跑的**：LLM 默认为规则规划器 `OfflineBackend`，它按确定规则产生
   `tool_use`，用于完整演示 AgentLoop / SubAgent 流程；配置真实 Key 后自动切换，
   接口形态与 OpenAI / Anthropic 一致（有集成测试覆盖）。
2. **嵌入是可切换的**：`auto` 优先用本地缓存的 BGE（512 维），否则用带符号哈希 + IDF
   的 1024 维离线替身。`fix_dim` 完整支持任意维度的补齐/截断，并有单元测试。
3. **微调粒度**：方案图 4 的"LoRA 微调 Qwen3.5-7B 当 Judge"需要 GPU 与真实数据；
   本工程用"规则 Judge + **真实** Embedding 三元损失训练"替代——三元损失的训练与
   Recall@k 评测都是真跑的，只是不涉及 7B。
4. **HyDE 为模板版**：真实系统用 LLM 生成假设文档，此处用模板构造，效果弱于真模型。

---

## 8. 目录结构

```
agentdoc-assistant/
├── agentdoc/                 # 主包（llm / loop / subagent / session / cli / config）
│   ├── rag/                  # 规范化 / 分块 / 嵌入（哈希+BGE）/ MQE+HyDE 检索
│   ├── memory/               # 三级记忆 / 评分 / 生命周期
│   ├── tools/                # 工具注册表与处理器
│   ├── context/              # 三级上下文压缩
│   └── mcp/                  # MCP server + client + host（stdio，纯标准库）
├── data/kb/                  # 示例知识库（8 篇 Markdown）
├── examples/                 # 示例外部 MCP server（文件系统）
├── experiments/              # exp1~exp5 + 汇总脚本
├── scripts/                   # verify_backends / run_with_llm / mcp_demo
├── tests/                    # 单元断言 + OpenAI 兼容集成测试 + MCP 集成测试
├── results/                  # 实验产物
└── run_all.sh
```
