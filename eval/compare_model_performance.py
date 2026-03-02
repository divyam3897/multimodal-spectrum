import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CONDITION_ORDER = ["Normal", "Image Shuffle", "Text Shuffle", "Random"]
NON_CATEGORY_KEYS = {
    "total",
    "correct",
    "total_questions",
    "total_correct",
    "overall_accuracy",
    "total_score",
    "is_multimodal",
    "multimodal_acc",
    "circular_accuracy",
    "circular_count",
    "accuracy",
    "total_count",
    "time",
    "model",
}
SKIP_DATASETS = {"infovqa", "docvqa", "mmvet", "stvqa", "vizwiz"}


def load_jsonl_file(file_path: str) -> List[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            data.append(json.loads(line))
    return data


def detect_available_datasets(base_dir: str, model_sizes: List[str]) -> List[str]:
    dataset_sets = []
    for model_size in model_sizes:
        answers_dir = os.path.join(base_dir, f"answers_{model_size}")
        if os.path.isdir(answers_dir):
            datasets = []
            for name in os.listdir(answers_dir):
                path = os.path.join(answers_dir, name)
                if os.path.isdir(path):
                    datasets.append(name)
            dataset_sets.append(set(datasets))
    if len(dataset_sets) == 0:
        return []
    return sorted(list(set.intersection(*dataset_sets)))


def parse_condition_from_filename(filename: str, prefix: str) -> str:
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 3 and parts[0] == prefix:
        suffix = parts[2].replace(".jsonl", "")
        if suffix.isdigit():
            return parts[1]
        return parts[1].replace(".jsonl", "")
    return os.path.basename(filename).split("_", 1)[1].replace(".jsonl", "")


def resolve_id_key(item: dict) -> str:
    if "question_id" in item:
        return "question_id"
    if "questionId" in item:
        return "questionId"
    if "idx" in item:
        return "idx"
    return "index"


def resolve_answer_key(item: dict) -> str:
    if "answer" in item:
        return "answer"
    if "prediction" in item:
        return "prediction"
    return "model_output"


def normalize_answer_for_voting(answer_text: str) -> str:
    text = str(answer_text).strip().lower()
    for prefix in ["answer is", "answer is:", "answer:", "the answer is"]:
        if prefix in text:
            text = text.split(prefix)[-1]
    text = text.strip().rstrip(".:,").lstrip("(").rstrip(")")
    return text


def create_ensemble_predictions(model_sizes: List[str], base_dir: str, dataset: str) -> str:
    conditions = set()
    for model_size in model_sizes:
        dataset_dir = os.path.join(base_dir, f"answers_{model_size}", dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for filename in os.listdir(dataset_dir):
            if filename.endswith(".jsonl"):
                parts = filename.split("_")
                if len(parts) >= 2 and parts[0] == model_size:
                    conditions.add(parse_condition_from_filename(filename, model_size))

    conditions = sorted(list(conditions))
    print(f"Detected conditions: {conditions}")
    ensemble_dir = os.path.join(base_dir, "answers_ensemble", dataset)
    os.makedirs(ensemble_dir, exist_ok=True)

    for condition in conditions:
        print(f"Processing condition: {condition}")
        predictions_by_qid = defaultdict(list)
        exemplar_by_qid = {}

        for model_size in model_sizes:
            first_path = os.path.join(base_dir, f"answers_{model_size}", dataset, f"{model_size}_{condition}.jsonl")
            second_path = os.path.join(base_dir, f"answers_{model_size}", dataset, f"{model_size}_{condition}_0.jsonl")
            target_path = first_path
            if not os.path.exists(first_path):
                target_path = second_path
            if not os.path.exists(target_path):
                print(f"File not found for {model_size}_{condition}")
                continue

            print(f"Found file: {target_path}")
            data = load_jsonl_file(target_path)
            for item in data:
                id_key = resolve_id_key(item)
                answer_key = resolve_answer_key(item)
                question_id = item[id_key]
                answer = item[answer_key]
                normalized = normalize_answer_for_voting(answer)
                predictions_by_qid[question_id].append(normalized)
                if question_id not in exemplar_by_qid:
                    exemplar_by_qid[question_id] = item.copy()

        ensemble_predictions = []
        for question_id in predictions_by_qid:
            pred_list = predictions_by_qid[question_id]
            unique_preds, counts = np.unique(pred_list, return_counts=True)
            majority_pred = unique_preds[int(np.argmax(counts))]
            row = exemplar_by_qid[question_id]
            answer_key = resolve_answer_key(row)
            row[answer_key] = majority_pred
            if answer_key != "answer":
                row["answer"] = majority_pred
            row["model_id"] = "ensemble"
            if "gt_answer" not in row and "ground_truth_answer" in row:
                row["gt_answer"] = row["ground_truth_answer"]
            ensemble_predictions.append(row)

        output_path = os.path.join(ensemble_dir, f"ensemble_{condition}.jsonl")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            for row in ensemble_predictions:
                file_obj.write(json.dumps(row) + "\n")
        print(f"Saved {len(ensemble_predictions)} ensemble predictions to: {output_path}")

    return ensemble_dir


def run_test_scripts_for_ensemble(base_dir: str, datasets: List[str]) -> None:
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
    for dataset in datasets:
        if dataset.lower() in SKIP_DATASETS:
            print(f"Skipping ensemble test for {dataset} (submission-only benchmark)")
            continue
        ensemble_dataset_dir = os.path.join(base_dir, "answers_ensemble", dataset)
        test_script = os.path.join(eval_dir, dataset, f"{dataset}_test.py")
        if not os.path.exists(test_script):
            print(f"No test script found for {dataset}, skipping ensemble eval")
            continue
        if not os.path.isdir(ensemble_dataset_dir):
            print(f"No ensemble predictions for {dataset}, skipping")
            continue
        print(f"Running {dataset}_test.py --compare_dir {ensemble_dataset_dir}")
        result = subprocess.run(
            [sys.executable, test_script, "--compare_dir", ensemble_dataset_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Warning: {dataset}_test.py returned exit code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
        else:
            print(f"  Ensemble comparison JSON generated for {dataset}")


def extract_score_from_category(score_value) -> float:
    if isinstance(score_value, dict):
        for key in ("accuracy", "score"):
            if key in score_value:
                return float(score_value[key])
        return 0.0
    if isinstance(score_value, (int, float)):
        return float(score_value)
    return 0.0


def load_comparison_data(answers_dir: str, dataset: str) -> dict:
    model_slug = os.path.basename(answers_dir).replace("answers_", "")
    dataset_dir = os.path.join(answers_dir, dataset)
    pattern = os.path.join(dataset_dir, f"{dataset}_comparison_*.json")
    matches = glob.glob(pattern)
    if len(matches) > 0:
        print(f"Loading comparison file: {matches[0]}")
        with open(matches[0], "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    print(f"No comparison JSON found for {model_slug}/{dataset}. Run the dataset's _test.py first.")
    return {"conditions": {}}


def scale_to_percent(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    max_value = float(np.nanmax(scaled.values))
    if max_value <= 1.5:
        scaled = scaled * 100.0
    return scaled


def radar_ymax(dataset: str) -> int:
    dataset_name = dataset.lower()
    if dataset_name == "scienceqa" or dataset_name == "mmmupro":
        return 60
    if dataset_name == "mmmu":
        return 80
    return 100


RADAR_TICKS = {60: [0, 20, 40, 60], 80: [0, 20, 40, 60, 80], 100: [0, 20, 40, 60, 80, 100]}


def set_plot_style() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 28


def create_radar_plot(data: pd.DataFrame, output_path: str, dataset: str) -> None:
    categories = []
    for category in data.index.tolist():
        category_text = str(category)
        if category_text.lower() not in NON_CATEGORY_KEYS:
            categories.append(category_text)
    if len(categories) == 0:
        print(f"No valid categories for radar: {output_path}")
        return

    data = data.loc[categories]
    data = scale_to_percent(data)
    series_names = data.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    set_plot_style()
    sns.set_theme(style="white", font_scale=1.2)
    palette = sns.color_palette("colorblind")
    color_map = {"Normal": "#de8f07", "Text Shuffle": "#019e73", "Image Shuffle": "#0173b2", "Random": "#d55e00"}
    _, axis = plt.subplots(figsize=(13.2, 13.2), subplot_kw=dict(polar=True))

    for i, series_name in enumerate(series_names):
        color = color_map[series_name] if series_name in color_map else palette[i % len(palette)]
        values = data[series_name].astype(float).values
        values = np.concatenate((values, [values[0]]))
        axis.plot(angles, values, linewidth=2.2, linestyle="-", label=series_name, color=color, marker=None)
        axis.fill(angles, values, alpha=0.15, color=color)

    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_thetagrids(np.degrees(angles[:-1]), categories)
    axis.set_xticklabels(categories, size=30, color="black")
    axis.tick_params(axis="x", pad=30)
    ymax = radar_ymax(dataset)
    ticks = RADAR_TICKS[ymax] if ymax in RADAR_TICKS else [0, 20, 40, 60, 80, 100]
    axis.set_ylim(0, ymax)
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{tick}%" for tick in ticks], fontsize=22, color="gray")
    axis.set_rlabel_position(135)
    axis.tick_params(axis="y", pad=-34)
    axis.yaxis.grid(True, linestyle="--", color="gray", alpha=0.35)
    axis.xaxis.grid(True, linestyle="-", color="lightgray", alpha=0.35)
    axis.spines["polar"].set_color("black")
    axis.spines["polar"].set_linewidth(1.0)
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(right=0.84, bottom=0.12)
    plt.savefig(output_path, format="pdf", dpi=400, bbox_inches="tight", pad_inches=0.15)
    plt.close()


def create_category_condition_bar_plot(data: pd.DataFrame, output_path: str) -> None:
    data = scale_to_percent(data)
    long_df = data.reset_index().rename(columns={"index": "Category"})
    long_df = long_df.melt(id_vars="Category", var_name="Condition", value_name="Score")
    set_plot_style()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(max(10, 1.2 * long_df["Category"].nunique()), 6.5))
    color_map = {"Normal": "#de8f07", "Text Shuffle": "#019e73", "Image Shuffle": "#0173b2", "Random": "#d55e00"}
    palette = [color_map[name] for name in CONDITION_ORDER]
    axis = sns.barplot(data=long_df, x="Category", y="Score", hue="Condition", hue_order=CONDITION_ORDER, palette=palette)
    legend = axis.get_legend()
    if legend is not None:
        legend.remove()
    axis.set_xlabel("")
    axis.set_ylabel("")
    plt.xticks(rotation=0)
    axis.tick_params(axis="x", labelsize=26)
    axis.tick_params(axis="y", labelsize=22)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=3, fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


def create_model_size_comparison_plots(model_data: Dict[str, dict], output_dir: str, dataset: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    conditions = set()
    for model_results in model_data.values():
        for condition in model_results["conditions"]:
            conditions.add(condition)
    conditions = sorted(list(conditions))
    if len(conditions) == 0:
        print(f"No conditions found for {dataset}")
        return

    first_cond = conditions[0]
    sample_cond = model_data[next(iter(model_data))]["conditions"][first_cond]
    sample_overall = sample_cond["overall_metrics"] if "overall_metrics" in sample_cond else {}
    if isinstance(sample_overall, dict):
        sample_metrics = list(sample_overall.keys())
    else:
        sample_metrics = ["accuracy"]
    for metric in sample_metrics:
        rows = []
        for model_size in model_data:
            model_results = model_data[model_size]
            for condition in conditions:
                if condition not in model_results["conditions"]:
                    continue
                cond_data = model_results["conditions"][condition]
                overall = cond_data["overall_metrics"] if "overall_metrics" in cond_data else {}
                if isinstance(overall, dict):
                    if metric not in overall:
                        continue
                    value = overall[metric]
                else:
                    value = overall
                display_size = "Ensemble" if model_size.lower() == "ensemble" else model_size.upper()
                rows.append({"Model Size": display_size, "Condition": condition, "Score": float(value)})
        if len(rows) == 0:
            continue
        df = pd.DataFrame(rows)
        if float(df["Score"].max()) <= 1.5:
            df["Score"] = df["Score"] * 100.0
        df["Condition"] = pd.Categorical(df["Condition"], categories=CONDITION_ORDER, ordered=True)
        model_order = [label for label in ["8B", "13B", "34B", "Ensemble"] if label in set(df["Model Size"].tolist())]
        df["Model Size"] = pd.Categorical(df["Model Size"], categories=model_order, ordered=True)
        df = df.sort_values(["Model Size", "Condition"])
        set_plot_style()
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(13.2, 6.8))
        color_map = {"Normal": "#de8f07", "Text Shuffle": "#019e73", "Image Shuffle": "#0173b2", "Random": "#d55e00"}
        palette = [color_map[name] for name in CONDITION_ORDER]
        axis = sns.barplot(data=df, x="Model Size", y="Score", hue="Condition", hue_order=CONDITION_ORDER, palette=palette)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        axis.set_xlabel("")
        axis.set_ylabel("")
        plt.xticks(rotation=0)
        axis.tick_params(axis="x", labelsize=26)
        axis.tick_params(axis="y", labelsize=22)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.0f", padding=3, fontsize=22)
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{dataset}_{metric}_model_size_comparison.pdf")
        plt.savefig(output_file, format="pdf", dpi=300, bbox_inches="tight")
        plt.close()

    for model_size in model_data:
        model_results = model_data[model_size]
        condition_to_categories = {}
        all_categories = set()
        for condition in CONDITION_ORDER:
            if condition not in model_results["conditions"]:
                continue
            cond_data = model_results["conditions"][condition]
            category_scores = cond_data["category_scores"] if "category_scores" in cond_data else {}
            if not category_scores:
                continue
            reduced_scores = {}
            for category in category_scores:
                if category.lower() in NON_CATEGORY_KEYS:
                    continue
                score_value = category_scores[category]
                extracted = extract_score_from_category(score_value)
                if extracted > 0:
                    reduced_scores[category] = extracted
            if len(reduced_scores) > 0:
                condition_to_categories[condition] = reduced_scores
                for category in reduced_scores:
                    all_categories.add(category)
        if len(condition_to_categories) == 0:
            continue
        all_categories = sorted(list(all_categories))
        cols = [condition for condition in CONDITION_ORDER if condition in condition_to_categories]
        df = pd.DataFrame(index=all_categories, columns=cols)
        for condition in condition_to_categories:
            for category in all_categories:
                value = 0.0
                if category in condition_to_categories[condition]:
                    value = condition_to_categories[condition][category]
                df.loc[category, condition] = value
        if len(all_categories) > 4:
            output_path = os.path.join(output_dir, f"{dataset}_{model_size}_category_condition_radar.pdf")
            create_radar_plot(df, output_path, dataset)
        else:
            output_path = os.path.join(output_dir, f"{dataset}_{model_size}_category_condition_bars.pdf")
            create_category_condition_bar_plot(df, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--output_dir", type=str, default="./model_size_comparison")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_sizes = ["8b", "13b", "34b"]
    if args.datasets is None:
        args.datasets = detect_available_datasets(args.base_dir, model_sizes)
        print(f"Auto-detected datasets: {args.datasets}")
    if len(args.datasets) == 0:
        print("No datasets found. Please check the base directory and model sizes.")
        return
    for dataset in args.datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        dataset_output_dir = os.path.join(args.output_dir, dataset)
        if not args.force and os.path.isdir(dataset_output_dir):
            has_pdf = any(name.endswith(".pdf") for name in os.listdir(dataset_output_dir))
            if has_pdf:
                print(f"Skipping dataset {dataset} (outputs already exist). Use --force to regenerate.")
                continue

        print(f"Creating ensemble predictions for {dataset}...")
        create_ensemble_predictions(model_sizes, args.base_dir, dataset)

        print(f"Running test script for ensemble/{dataset}...")
        run_test_scripts_for_ensemble(args.base_dir, [dataset])

        model_data = {}
        for model_size in model_sizes:
            answers_dir = os.path.join(args.base_dir, f"answers_{model_size}")
            if not os.path.isdir(answers_dir):
                print(f"Warning: Answers directory {answers_dir} does not exist")
                continue
            model_data[model_size] = load_comparison_data(answers_dir, dataset)

        ensemble_answers_dir = os.path.join(args.base_dir, "answers_ensemble")
        if os.path.isdir(ensemble_answers_dir):
            model_data["ensemble"] = load_comparison_data(ensemble_answers_dir, dataset)

        available_keys = [key for key in model_data if len(model_data[key]["conditions"]) > 0]
        if len(available_keys) == 0:
            print(f"No comparison data found for {dataset}; skipping.")
            continue

        if dataset.lower() in SKIP_DATASETS:
            print(f"Dataset {dataset} is submission-only or lacks numeric metrics; skipping plotting.")
            continue

        create_model_size_comparison_plots(
            {k: model_data[k] for k in available_keys},
            dataset_output_dir,
            dataset,
        )
        print(f"Created comparison plots in: {dataset_output_dir}")


if __name__ == "__main__":
    main()
