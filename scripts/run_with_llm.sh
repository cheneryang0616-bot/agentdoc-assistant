#!/usr/bin/env bash
# 从 ~/.agentdoc.env 读取 LLM / 嵌入配置后启动 CLI（避免把 Key 写进仓库）
#
# 准备：
#   cp .env.example ~/.agentdoc.env
#   vim ~/.agentdoc.env          # 填入 DEEPSEEK_API_KEY=sk-xxxx
#
# 运行：
#   bash scripts/run_with_llm.sh --demo
#   bash scripts/run_with_llm.sh --query "JVM 是什么？"
#   bash scripts/run_with_llm.sh            # 交互模式
set -e
cd "$(dirname "$0")/.."

if [ -f "$HOME/.agentdoc.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.agentdoc.env"
else
  echo "[提示] 未找到 ~/.agentdoc.env，将使用默认配置（无 Key 时回退离线规划器）"
  echo "       可执行: cp .env.example ~/.agentdoc.env 后填入 Key"
fi

exec python3 -m agentdoc.cli "$@"
