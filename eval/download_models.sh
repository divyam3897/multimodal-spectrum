#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_TYPE="${1:-}"
MODEL_SIZE="${2:-}"

if [ -z "$MODEL_TYPE" ] || [ -z "$MODEL_SIZE" ]; then
  echo "Usage: $0 <model_type> <model_size>"
  echo "  cambrian:  8b, 13b, 34b"
  echo "  qwen2_5:   7b"
  echo "  qwen3:     8b"
  echo "  llava-next: 7b-mistral"
  exit 1
fi

case "$MODEL_TYPE" in
  "cambrian")
    MODEL_DIR="${SCRIPT_DIR}/cambrian-${MODEL_SIZE}"
    case "$MODEL_SIZE" in
      "8b")  HF_REPO="nyu-visionx/cambrian-8b" ;;
      "13b") HF_REPO="nyu-visionx/cambrian-13b" ;;
      "34b") HF_REPO="nyu-visionx/cambrian-34b" ;;
      *)
        echo "Unsupported cambrian size: $MODEL_SIZE (use 8b, 13b, 34b)"
        exit 1
        ;;
    esac
    ;;
  "qwen2_5")
    MODEL_DIR="${SCRIPT_DIR}/qwen2.5-vl-${MODEL_SIZE}"
    case "$MODEL_SIZE" in
      "7b") HF_REPO="Qwen/Qwen2.5-VL-7B-Instruct" ;;
      *)
        echo "Unsupported qwen2_5 size: $MODEL_SIZE (use 7b)"
        exit 1
        ;;
    esac
    ;;
  "qwen3")
    MODEL_DIR="${SCRIPT_DIR}/qwen3-vl-${MODEL_SIZE}"
    case "$MODEL_SIZE" in
      "8b") HF_REPO="Qwen/Qwen3-VL-8B-Instruct" ;;
      *)
        echo "Unsupported qwen3 size: $MODEL_SIZE (use 8b)"
        exit 1
        ;;
    esac
    ;;
  "llava-next")
    case "$MODEL_SIZE" in
      "7b-mistral")
        MODEL_DIR="${SCRIPT_DIR}/llava-next-7b-mistral"
        HF_REPO="llava-hf/llava-v1.6-mistral-7b-hf"
        ;;
      *)
        echo "Unsupported llava-next size: $MODEL_SIZE (use 7b-mistral)"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported model_type: $MODEL_TYPE"
    echo "Use: cambrian, qwen2_5, qwen3, llava-next"
    exit 1
    ;;
esac

echo "Downloading ${MODEL_TYPE}-${MODEL_SIZE} from ${HF_REPO} -> ${MODEL_DIR}"
mkdir -p "$MODEL_DIR"

if command -v huggingface-cli &> /dev/null; then
  huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
else
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${HF_REPO}', local_dir='${MODEL_DIR}', local_dir_use_symlinks=False)
print('Done.')
"
fi

echo "Downloaded to $MODEL_DIR"
