#!/bin/bash
#SBATCH --job-name=spectrum_multi_benchmark
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=radiology
#SBATCH --output=spectrum_multi_benchmark_%j.out

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  DIR="${SLURM_SUBMIT_DIR}"
fi

if [ $# -eq 2 ]; then
  MODEL_TYPE="cambrian"
  MODEL_SIZE="$1"
  BENCHMARKS="$2"
elif [ $# -eq 3 ]; then
  MODEL_TYPE="$1"
  MODEL_SIZE="$2"
  BENCHMARKS="$3"
else
  echo "Usage: sbatch $0 [MODEL_TYPE] <MODEL_SIZE> <BENCHMARKS>. See README.md."
  exit 1
fi

case "$MODEL_TYPE" in
  "cambrian")
    case "$MODEL_SIZE" in
      "8b")  CONV_MODE="llama_3" ;;
      "13b") CONV_MODE="vicuna_v1" ;;
      "34b") CONV_MODE="chatml_direct" ;;
      *)
        echo "Unsupported MODEL_SIZE for cambrian: $MODEL_SIZE (use 8b, 13b, 34b)"
        exit 1
        ;;
    esac
    MODEL_DIR="${DIR}/cambrian-${MODEL_SIZE}/"
    ANSWERS_LABEL="${MODEL_SIZE}"
    FILE_LABEL="${MODEL_SIZE}"
    ;;
  "qwen2_5")
    case "$MODEL_SIZE" in
      "7b") CONV_MODE="qwen_2" ;;
      *)
        echo "Unsupported MODEL_SIZE for qwen2_5: $MODEL_SIZE (use 7b)"
        exit 1
        ;;
    esac
    MODEL_DIR="${DIR}/qwen2.5-vl-${MODEL_SIZE}/"
    ANSWERS_LABEL="qwen2_5_${MODEL_SIZE}"
    FILE_LABEL="qwen2_5_${MODEL_SIZE}"
    ;;
  "qwen3")
    case "$MODEL_SIZE" in
      "8b") CONV_MODE="qwen_3" ;;
      *)
        echo "Unsupported MODEL_SIZE for qwen3: $MODEL_SIZE (use 8b)"
        exit 1
        ;;
    esac
    MODEL_DIR="${DIR}/qwen3-vl-${MODEL_SIZE}/"
    ANSWERS_LABEL="qwen3_${MODEL_SIZE}"
    FILE_LABEL="qwen3_${MODEL_SIZE}"
    ;;
  "llava-next")
    case "$MODEL_SIZE" in
      "7b-mistral")
        CONV_MODE="llama_3"
        MODEL_DIR="${DIR}/llava-next-7b-mistral/"
        ;;
      *)
        echo "Unsupported MODEL_SIZE for llava-next: $MODEL_SIZE (use 7b-mistral)"
        exit 1
        ;;
    esac
    ANSWERS_LABEL="llava-next_${MODEL_SIZE}"
    FILE_LABEL="llava-next_${MODEL_SIZE}"
    ;;
  *)
    echo "Unsupported MODEL_TYPE: $MODEL_TYPE"
    echo "Use: cambrian, qwen2_5, qwen3, llava-next"
    exit 1
    ;;
esac

module purge
module load cuda/11.8

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

VENV_ACTIVATE="${DIR}/.venv/bin/activate"
[ -f "$VENV_ACTIVATE" ] || VENV_ACTIVATE="${DIR}/../.venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
  echo "Using Python: $(which python)"
else
  echo "Warning: ${VENV_ACTIVATE} not found; falling back to cambrian_fork env"
  source /gpfs/data/chopralab/dm5182/cambrian_fork/eval/cambrian_env/bin/activate
fi

if [ "$MODEL_TYPE" = "cambrian" ] && [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Model not found at ${MODEL_DIR}, downloading nyu-visionx/cambrian-${MODEL_SIZE} from HuggingFace..."
  mkdir -p "${MODEL_DIR}"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nyu-visionx/cambrian-${MODEL_SIZE}', local_dir='${MODEL_DIR}', local_dir_use_symlinks=False)
print('Download complete.')
"
fi

if [ "$MODEL_TYPE" != "cambrian" ] && [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model directory not found: $MODEL_DIR"
  echo "Download with: ./download_models.sh <model_type> <model_size>. See README.md."
  exit 1
fi

ALL_BENCHMARKS="mme mmmu mmmupro mmbench_en mmbench_cn gqa textvqa vizwiz coco pope ade ai2d blink chartqa mathvista mmstar mmvp ocrbench omni qbench realworldqa scienceqa seed vstar"

if [ "$BENCHMARKS" = "all" ]; then
  BENCHMARK_LIST="$ALL_BENCHMARKS"
else
  BENCHMARK_LIST=$(echo "$BENCHMARKS" | tr ',' ' ')
fi

echo "Running benchmarks: $BENCHMARK_LIST"
echo "Model type: $MODEL_TYPE"
echo "Model size: $MODEL_SIZE"
echo "Conversation mode: $CONV_MODE"
echo "Model directory: $MODEL_DIR"
echo "Answers directory: ${DIR}/answers_${ANSWERS_LABEL}/"

EVAL_EXTRA_ARGS=""
[ "$MODEL_TYPE" != "cambrian" ] && EVAL_EXTRA_ARGS="--model_type ${MODEL_TYPE}"

for BENCHMARK in $BENCHMARK_LIST; do
  echo "Starting benchmark: $BENCHMARK"

  BENCH_DIR="${DIR}/eval/${BENCHMARK}"
  ANSWERS_DIR="${DIR}/answers_${ANSWERS_LABEL}/${BENCHMARK}"

  if [ ! -d "$BENCH_DIR" ]; then
    echo "Warning: Evaluation directory $BENCH_DIR does not exist, skipping $BENCHMARK"
    continue
  fi

  if [ ! -f "${BENCH_DIR}/${BENCHMARK}_eval.py" ]; then
    echo "Warning: Evaluation script ${BENCH_DIR}/${BENCHMARK}_eval.py does not exist, skipping $BENCHMARK"
    continue
  fi

  mkdir -p "$ANSWERS_DIR"

  echo "Running normal evaluation for $BENCHMARK..."
  python "${BENCH_DIR}/${BENCHMARK}_eval.py" \
    --model_path "${MODEL_DIR}" \
    --answers_file "${ANSWERS_DIR}/${FILE_LABEL}_nrm.jsonl" \
    --conv_mode "${CONV_MODE}" \
    $EVAL_EXTRA_ARGS
  echo "FINISHED NORMAL EVALUATION FOR $BENCHMARK"

  echo "Running text shuffle evaluation for $BENCHMARK..."
  python "${BENCH_DIR}/${BENCHMARK}_eval.py" \
    --model_path "${MODEL_DIR}" \
    --answers_file "${ANSWERS_DIR}/${FILE_LABEL}_txt.jsonl" \
    --text_shuffle --conv_mode "${CONV_MODE}" \
    $EVAL_EXTRA_ARGS
  echo "FINISHED TEXT SHUFFLE EVALUATION FOR $BENCHMARK"

  echo "Running image shuffle evaluation for $BENCHMARK..."
  python "${BENCH_DIR}/${BENCHMARK}_eval.py" \
    --model_path "${MODEL_DIR}" \
    --answers_file "${ANSWERS_DIR}/${FILE_LABEL}_img.jsonl" \
    --image_shuffle --conv_mode "${CONV_MODE}" \
    $EVAL_EXTRA_ARGS
  echo "FINISHED IMAGE SHUFFLE EVALUATION FOR $BENCHMARK"

  echo "Running random shuffle evaluation for $BENCHMARK..."
  python "${BENCH_DIR}/${BENCHMARK}_eval.py" \
    --model_path "${MODEL_DIR}" \
    --answers_file "${ANSWERS_DIR}/${FILE_LABEL}_rdm.jsonl" \
    --text_shuffle --image_shuffle --conv_mode "${CONV_MODE}" \
    $EVAL_EXTRA_ARGS
  echo "FINISHED RANDOM SHUFFLE EVALUATION FOR $BENCHMARK"

  echo "Completed all evaluations for $BENCHMARK"
done

echo "All benchmarks completed."
