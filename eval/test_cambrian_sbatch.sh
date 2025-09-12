#!/bin/bash
#SBATCH --job-name=cambrian_multi_benchmark
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=radiology
#SBATCH --output=cambrian_multi_benchmark_%j.out

# --- Base Directory ---
BASE_DIR="/gpfs/data/chopralab/dm5182/cambrian_fork/eval"

# --- Argument Parsing ---
MODEL_SIZE="$1"
BENCHMARKS="$2"

if [ -z "$MODEL_SIZE" ] || [ -z "$BENCHMARKS" ]; then
  echo "Usage: sbatch $0 <MODEL_SIZE> <BENCHMARKS>"
  echo "Examples:"
  echo "  sbatch $0 8b mme,mmmu,mmmupro"
  echo "  sbatch $0 13b mme"
  echo "  sbatch $0 34b all"
  echo ""
  echo "Available benchmarks:"
  echo "  mme, mmmu, mmmupro, mmbench_en, mmbench_cn, gqa, textvqa, vizwiz,"
  echo "  coco, pope, ade, ai2d, blink, chartqa, docvqa, infovqa, mathvista,"
  echo "  mmstar, mmvet, mmvp, ocrbench, omni, qbench, realworldqa, scienceqa,"
  echo "  seed, stvqa, synthdog, vstar"
  exit 1
fi

# Set conversation mode based on MODEL_SIZE
case "$MODEL_SIZE" in
  "8b")
    CONV_MODE="llama_3"
    ;;
  "13b")
    CONV_MODE="vicuna_v1"
    ;;
  "34b")
    CONV_MODE="chatml_direct"
    ;;
  *)
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE"
    exit 1
    ;;
esac

module purge
module load cuda/11.8
source /gpfs/data/chopralab/dm5182/cambrian_fork/eval/cambrian_env/bin/activate

MODEL_DIR="${BASE_DIR}/cambrian-${MODEL_SIZE}/"

# Define all available benchmarks
# need submissions: infoqa, docqa, mmvet
# ALL_BENCHMARKS="mme mmmu mmmupro mmbench_en mmbench_cn gqa textvqa vizwiz coco pope ade ai2d blink chartqa docvqa infovqa mmstar mmvet mmvp ocrbench omni qbench realworldqa scienceqa seed stvqa synthdog vstar mathvista"
# ALL_BENCHMARKS="qbench omni textvqa realworldqa scienceqa ocrbench pope"
ALL_BENCHMARKS="ocrbench scienceqa ai2d"
# ALL_BENCHMARKS="mathvista"s

if [ "$BENCHMARKS" = "all" ]; then
    BENCHMARK_LIST="$ALL_BENCHMARKS"
else
    # Convert comma-separated list to space-separated
    BENCHMARK_LIST=$(echo "$BENCHMARKS" | tr ',' ' ')
fi

echo "Running benchmarks: $BENCHMARK_LIST"
echo "Model size: $MODEL_SIZE"
echo "Conversation mode: $CONV_MODE"

# --- Run Evaluations for Each Benchmark ---
for BENCHMARK in $BENCHMARK_LIST; do
    echo "=========================================="
    echo "Starting benchmark: $BENCHMARK"
    echo "=========================================="
    
    EVAL_DIR="${BASE_DIR}/eval/${BENCHMARK}"
    ANSWERS_DIR="${BASE_DIR}/answers_${MODEL_SIZE}/${BENCHMARK}"
    
    # Check if evaluation directory exists
    if [ ! -d "$EVAL_DIR" ]; then
        echo "Warning: Evaluation directory $EVAL_DIR does not exist, skipping $BENCHMARK"
        continue
    fi
    
    # Check if evaluation script exists
    if [ ! -f "${EVAL_DIR}/${BENCHMARK}_eval.py" ]; then
        echo "Warning: Evaluation script ${EVAL_DIR}/${BENCHMARK}_eval.py does not exist, skipping $BENCHMARK"
        continue
    fi
    
    mkdir -p "$ANSWERS_DIR"
    
    # Run normal evaluation
    echo "Running normal evaluation for $BENCHMARK..."
    python ${EVAL_DIR}/${BENCHMARK}_eval.py \
      --model_path ${MODEL_DIR} \
      --answers_file ${ANSWERS_DIR}/${MODEL_SIZE}_nrm.jsonl \
      --conv_mode ${CONV_MODE}
    echo "FINISHED NORMAL EVALUATION FOR $BENCHMARK"
    
    # Run text shuffle evaluation
    echo "Running text shuffle evaluation for $BENCHMARK..."
    python ${EVAL_DIR}/${BENCHMARK}_eval.py \
      --model_path ${MODEL_DIR} \
      --answers_file ${ANSWERS_DIR}/${MODEL_SIZE}_txt.jsonl \
      --text_shuffle --conv_mode ${CONV_MODE}
    echo "FINISHED TEXT SHUFFLE EVALUATION FOR $BENCHMARK"
    
    # Run image shuffle evaluation
    echo "Running image shuffle evaluation for $BENCHMARK..."
    python ${EVAL_DIR}/${BENCHMARK}_eval.py \
      --model_path ${MODEL_DIR} \
      --answers_file ${ANSWERS_DIR}/${MODEL_SIZE}_img.jsonl \
      --image_shuffle --conv_mode ${CONV_MODE}
    echo "FINISHED IMAGE SHUFFLE EVALUATION FOR $BENCHMARK"
    
    # Run random shuffle evaluation
    echo "Running random shuffle evaluation for $BENCHMARK..."
    python ${EVAL_DIR}/${BENCHMARK}_eval.py \
      --model_path ${MODEL_DIR} \
      --answers_file ${ANSWERS_DIR}/${MODEL_SIZE}_rdm.jsonl \
      --text_shuffle --image_shuffle --conv_mode ${CONV_MODE}
    echo "FINISHED RANDOM SHUFFLE EVALUATION FOR $BENCHMARK"
    
    echo "=========================================="
    echo "Completed all evaluations for $BENCHMARK"
    echo "=========================================="
done

echo "=========================================="
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "=========================================="
