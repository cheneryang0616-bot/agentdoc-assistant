#!/usr/bin/env bash
# 一键运行全部复现实验并汇总结果
set -e
cd "$(dirname "$0")"

echo "==== 0. 单元测试 ===="
python3 tests/test_core.py

echo; echo "==== 0b. OpenAI 兼容后端集成测试（本地 mock，无需 Key）===="
python3 tests/test_openai_backend.py

echo; echo "==== 0c. MCP 集成测试（stdio）===="
python3 tests/test_mcp.py

echo; echo "==== 0d. MCP Host 集成测试（多 server）===="
python3 tests/test_mcp_host.py

echo; echo "==== 1. AgentLoop vs 单次 RAG ===="
python3 experiments/exp1_agent_loop.py

echo; echo "==== 2. 三级上下文压缩 ===="
python3 experiments/exp2_context_compression.py

echo; echo "==== 3. 记忆生命周期管理 ===="
python3 experiments/exp3_memory_lifecycle.py

echo; echo "==== 4. 数据蒸馏 + Embedding 微调 ===="
python3 experiments/exp4_finetune_embedding.py

echo; echo "==== 5. 真实嵌入 vs 哈希嵌入 ===="
python3 experiments/exp5_embedder_compare.py

echo; echo "==== 6. 汇总 RESULTS.md ===="
python3 experiments/summarize.py

echo; echo "==== 7. 后端自检 ===="
python3 scripts/verify_backends.py

echo; echo "==== 8. 交互式 Agent Demo ===="
python3 -m agentdoc.cli --demo

echo; echo "==== 9. MCP 端到端演示 ===="
python3 scripts/mcp_demo.py

echo; echo "==== 10. MCP Host 演示（多 server）===="
python3 scripts/mcp_host_demo.py

echo; echo "全部完成。结果见 results/RESULTS.md"
