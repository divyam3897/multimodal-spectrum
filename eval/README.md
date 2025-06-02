# MLLM Eval

> [!TIP]
> ***See the [slurm/](slurm/) dir for instructions on running evaluations in parallel on HPC clusters using Slurm.***


## Overview

This directory contains the evaluation scripts and benchmarks for the Cambrian-1 multimodal language model. It includes a wide range of benchmarks to assess the model's performance across various tasks and domains.

## Benchmarks

The following benchmarks are included:

1. GQA
2. VizWiz
3. ScienceQA
4. TextVQA
5. POPE
6. MME
7. MMBench (English and Chinese)
8. SEED
9. MMMU
10. MathVista
11. AI2D
12. ChartQA
13. DocVQA
14. InfoVQA
15. STVQA
16. OCRBench
17. MMStar
18. RealWorldQA
19. SynthDog
20. QBench
21. BLINK
22. MMVP
23. VStar
24. ADE
25. OMNI
26. COCO

Each benchmark has its own subdirectory under [`eval/`](eval/) containing:
* an evaluation script (`*_eval.py`) — generates answers and saves them in a `.jsonl` file
* a testing script (`*_test.py`) — reads in the `.jsonl` answers file, performs any necessary post-processing and matching with ground truth, and appends the results to a common `.csv` file (for the benchmark) keyed on the `model_id` and `time` of the evaluation

## Setup

1. Ensure you have the required dependencies installed. You can find these in the [`requirements.txt`](requirements.txt) file in this subdir. You also need the Cambrian codebase installed in the parent directory.

2. The datasets will be downloaded automatically when you run the evaluation scripts.

## Shuffling Modality Usage
### Different language models
To run benchmarks with different modalities shuffled, use the [test_cambrian_sbatch.sh](./test_cambrian_sbatch.sh) script. 

To use it, run the script with the first argument telling which language model to use, with choices between `8b` (llama_3), `13b` (vicuna_v1), and `34b` (chatml_direct). The second argument decides which benchmark you would like to use, with options of `mme`, `mathvista`, `mmmu`, `mmmupro`, `ade`, `chartqa`, `gqa`,  `mmbench_cn`, `mmbench_en`, `mmstar`, `mmvp`.

Basic usage example looks like:

```bash
bash test_cambrian_sbatch.sh mmbench_cn 34b
```

Running this saves in the `answers` directory a new folder that contains the benchmark as well as answer files for each language model and each modality being switched on or off. Run the `eval/<benchmark>/<benchmark>_test.py` script on each of these answer files in order to generate the performance metrics.

This script is designed to run on NYU's BigPurple cluster, so modification of the scripts arguments internally for cluster and resources might be necessary. 


### Different image processors
The only benchmarks that support evaluation using different image processors is the main four from the paper; `mme`, `mmmupro`, `mmmu`, `mathvista`. There is no sbatch script for those unfortunately, but the python script for each is under their respective `eval/<benchmark>` folders, with the names of `<benchmark>_eval_ipc.py` and `<benchmark>_eval_img_ipc.py`. The `*img_ipc.py` file is for the shuffled text and maintained image and `*ipc.py` is for maintained text and image. 

Running these scripts will produce 4 files in your designated folder which contain a run where only one image processor is embedding the image. `*_1.jsonl` will correspond to answers from just the siglip/CLIP-ViT-SO400M-14-384 processor, `*_2.jsonl` will correspond to answers from just the openai/clip-vit-large-patch14-336 processor, `*_3.jsonl` will correspond to answers from just the facebook/dinov2-giant-res378 processor, and `*_4.jsonl` will correspond to answers from just the clip-convnext-XXL-multi-stage processor.

In order to specify the other arguments, just looking at how the [test_cambrian_sbatch.sh](./test_cambrian_sbatch.sh) shell script runs the regular evaluation script. The only arguments you do not need to worry about setting here from the shell script are `--text_shuffle` and `--image_shuffle`; as the shuffled modalities for each of the two image processor swap scripts are already set in stone (explained above).
## Usage

To run evaluations, use the [`run_benchmark.sh`](scripts/run_benchmark.sh) script in the [`scripts/`](scripts/) directory. Here's the basic usage:

```bash
bash scripts/run_benchmark.sh --benchmark <benchmark_name> --ckpt <path_to_checkpoint> --conv_mode <conversation_mode>
```

For example:

```bash
bash scripts/run_benchmark.sh --benchmark mmmu --ckpt /path/to/cambrian/checkpoint --conv_mode llama_3
```

or using the [`nyu-visionx/cambrian-8b`](https://huggingface.co/nyu-visionx/cambrian-8b) HF model:

```bash
bash scripts/run_benchmark.sh --benchmark mmmu --ckpt nyu-visionx/cambrian-8b --conv_mode llama_3
```



### Running All Benchmarks

To ***sequentially*** run all benchmarks for a single checkpoint, use the [`run_all_benchmarks.sh`](scripts/run_all_benchmarks.sh) script:

```bash
bash scripts/run_all_benchmarks.sh /path/to/cambrian/checkpoint llama_3
```

This script will run all implemented benchmarks and save progress in a checkpoint file.

> [!TIP]
> See the [slurm/](slurm/) dir for instructions on running evaluations in parallel on HPC clusters using Slurm.

## Tabulating Results

After running the evaluations, you can use the [`tabulate.py`](scripts/tabulate.py) script to compile the results:

```bash
python scripts/tabulate.py --eval_dir eval --experiment_csv experiments.csv --out_pivot pivot.xlsx --out_all_results all_results.csv
```

This will generate:
- A long CSV file (`all_results.csv`) with all compiled results
- An Excel/CSV file (`pivot.xlsx`/`pivot.csv`) with the final metrics for each benchmark as columns and each model evaluated as a row

## Contributing

If you want to add a new benchmark or improve existing ones, please follow the structure of the existing benchmark directories and update the `run_all_benchmarks.sh` script accordingly.

## TODO:

- [ ] add GPT / LLM evaluation option to grade answers instead of the manual / fuzzy matching currently used in the `*_test.py` files