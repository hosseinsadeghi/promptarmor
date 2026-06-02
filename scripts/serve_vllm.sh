#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-models/Qwen3-32B}"   # local dir, or an HF id like Qwen/Qwen3-32B
PORT="${2:-8000}"
# Once weights are staged locally, force offline so nothing hits the network at inference:
# export HF_HUB_OFFLINE=1
exec vllm serve "$MODEL" --enable-auto-tool-choice --reasoning-parser qwen3 --port "$PORT"
