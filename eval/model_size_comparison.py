import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union
import shutil
from collections import defaultdict
import argparse
import random
from datetime import datetime

EVAL_TYPE_MAPPING = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}

REVERSE_EVAL_MAPPING = {item: key for key, values in EVAL_TYPE_MAPPING.items() for item in values}

def load_jsonl_file(file_path: str) -> List[dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def detect_available_datasets(base_dir: str, model_sizes: List[str]) -> List[str]:
    dataset_sets = []
    for size in model_sizes:
        answers_dir = os.path.join(base_dir, f'answers_{size}')
        if os.path.exists(answers_dir):
            datasets = set([
                d for d in os.listdir(answers_dir)
                if os.path.isdir(os.path.join(answers_dir, d))
            ])
            dataset_sets.append(datasets)
    if not dataset_sets:
        return []
    available_datasets = set.intersection(*dataset_sets)
    return sorted(list(available_datasets))

def create_ensemble_predictions(model_sizes: List[str], base_dir: str, dataset: str) -> str:
    conditions = set()
    for size in model_sizes:
        dataset_dir = os.path.join(base_dir, f'answers_{size}/{dataset}')
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                if file.endswith('.jsonl'):
                    parts = file.split('_')
                    if len(parts) >= 2 and parts[0] == size:
                        if len(parts) >= 3 and parts[2].replace('.jsonl', '').isdigit():
                            condition = parts[1]
                        else:
                            condition = parts[1].replace('.jsonl', '')
                        conditions.add(condition)
    
    conditions = list(conditions)
    print(f"Detected conditions: {conditions}")
    
    ensemble_dir = os.path.join(base_dir, 'answers_ensemble', dataset)
    os.makedirs(ensemble_dir, exist_ok=True)
    
    for condition in conditions:
        print(f"Processing condition: {condition}")
        predictions = defaultdict(list)
        
        for size in model_sizes:
            jsonl_path = os.path.join(base_dir, f'answers_{size}/{dataset}/{size}_{condition}.jsonl')
            if not os.path.exists(jsonl_path):
                jsonl_path = os.path.join(base_dir, f'answers_{size}/{dataset}/{size}_{condition}_0.jsonl')
            
            if os.path.exists(jsonl_path):
                print(f"Found file: {jsonl_path}")
                data = load_jsonl_file(jsonl_path)
                for item in data:
                    q_id = item.get('question_id') or item.get('questionId') or item.get('idx') or item.get('index')
                    if q_id is not None:
                        answer = item.get('answer') or item.get('prediction') or item.get('model_output')
                        if answer is not None:
                            predictions[q_id].append(answer)
            else:
                print(f"File not found for {size}_{condition} (tried both patterns)")
        
        ensemble_predictions = []
        for q_id, preds in predictions.items():
            if preds:
                unique_preds, counts = np.unique(preds, return_counts=True)
                majority_pred = unique_preds[np.argmax(counts)]
                
                base_pred = None
                for size in model_sizes:
                    jsonl_path = os.path.join(base_dir, f'answers_{size}/{dataset}/{size}_{condition}.jsonl')
                    if not os.path.exists(jsonl_path):
                        jsonl_path = os.path.join(base_dir, f'answers_{size}/{dataset}/{size}_{condition}_0.jsonl')
                    
                    if os.path.exists(jsonl_path):
                        data = load_jsonl_file(jsonl_path)
                        for item in data:
                            item_q_id = item.get('question_id') or item.get('questionId') or item.get('idx') or item.get('index')
                            if item_q_id == q_id:
                                base_pred = item.copy()
                                break
                    if base_pred:
                        break
                
                if base_pred:
                    if 'answer' in base_pred:
                        base_pred['answer'] = majority_pred
                    elif 'prediction' in base_pred:
                        base_pred['prediction'] = majority_pred
                    elif 'model_output' in base_pred:
                        base_pred['model_output'] = majority_pred
                        base_pred['answer'] = majority_pred
                    else:
                        base_pred['answer'] = majority_pred
                    if 'gt_answer' not in base_pred and 'ground_truth_answer' in base_pred:
                        base_pred['gt_answer'] = base_pred['ground_truth_answer']
                    ensemble_predictions.append(base_pred)
        
        output_path = os.path.join(ensemble_dir, f'ensemble_{condition}.jsonl')
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in ensemble_predictions:
                f.write(json.dumps(pred) + '\n')
        
        print(f"Saved {len(ensemble_predictions)} ensemble predictions to: {output_path}")
    
    return ensemble_dir



def calculate_metrics(data: List[dict]) -> dict:
    total = len(data)
    correct = sum(1 for item in data if item['answer'] == item['gt_answer'])
    accuracy = correct / total if total > 0 else 0
    
    category_scores = defaultdict(list)
    for item in data:
        category = item.get('category', 'unknown')
        category_scores[category].append(1 if item['answer'] == item['gt_answer'] else 0)
    
    category_metrics = {}
    for category, scores in category_scores.items():
        category_metrics[category] = sum(scores) / len(scores)
    
    return {
        'overall_metrics': {
            'overall_accuracy': accuracy,
            'total_score': accuracy
        },
        'category_scores': category_metrics
    }

def calculate_mme_metrics(data: List[dict]) -> dict:
    category_stats = {}
    for i, item in enumerate(data):
        category = item.get('category')
        if not category: continue
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'acc_plus_running_total': 0, 'total': 0}
        
        gt_answer = str(item.get('gt_answer', '')).lower().strip().rstrip('.')
        answer = str(item.get('answer', '')).lower().strip().rstrip('.')
        
        stats = category_stats[category]
        stats['total'] += 1
        is_correct = (answer == gt_answer)
        
        if is_correct:
            stats['correct'] += 1
            if i % 2 == 0:
                stats['acc_plus_running_total'] += 1
            elif stats['acc_plus_running_total'] % 2 == 1:
                stats['acc_plus_running_total'] += 1
        elif i % 2 == 1 and stats['acc_plus_running_total'] % 2 == 1:
            stats['acc_plus_running_total'] -= 1
            
    category_scores = {}
    total_correct, total_count, total_acc_plus = 0, 0, 0
    for category, stats in category_stats.items():
        total = stats['total']
        if total == 0: continue
        accuracy = stats['correct'] / total
        acc_plus = stats['acc_plus_running_total'] / total
        category_scores[category] = {
            'type': REVERSE_EVAL_MAPPING.get(category, 'Unknown'),
            'accuracy': accuracy, 'accuracy_plus': acc_plus,
            'score': 100 * (accuracy + acc_plus),
            'correct': stats['correct'], 'total': total,
        }
        total_correct += stats['correct']
        total_count += total
        total_acc_plus += stats['acc_plus_running_total']

    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0
    overall_acc_plus = (total_acc_plus / total_count) if total_count > 0 else 0
    perception_score = sum(cs['score'] for cs in category_scores.values() if cs['type'] == 'Perception')
    cognition_score = sum(cs['score'] for cs in category_scores.values() if cs['type'] == 'Cognition')
    
    return {
        'overall_metrics': {
            'total_score': perception_score + cognition_score,
            'perception_score': perception_score, 'cognition_score': cognition_score,
            'overall_accuracy': overall_accuracy, 'overall_accuracy_plus': overall_acc_plus,
            'total_questions': total_count, 'total_correct': total_correct,
        },
        'category_scores': category_scores
    }

data_metric_fn = {
    'mme': calculate_mme_metrics,
}

def load_comparison_data(answers_dir: str, dataset: str) -> Union[dict, None]:
    json_pattern = os.path.join(answers_dir, dataset, f'{dataset}_comparison_*.json')
    json_files = glob.glob(json_pattern)
    model_size_slug = os.path.basename(answers_dir).replace('answers_', '')
    matching_file = None
    for f in json_files:
        if f'{dataset}_comparison_{model_size_slug}' in os.path.basename(f):
            matching_file = f
            break
    if matching_file:
        print(f"Loading existing comparison file: {matching_file}")
        with open(matching_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"No pre-computed file found for {model_size_slug}. Generating from .jsonl files...")
    dataset_dir = os.path.join(answers_dir, dataset)
    if not os.path.exists(dataset_dir):
        print(f"No dataset directory found at {dataset_dir}")
        return None
    conditions = {}
    condition_display_names = {
        'nrm': 'Normal',
        'img': 'Text Shuffle',
        'txt': 'Image Shuffle',
        'rdm': 'Random'
    }
    file_prefix = 'ensemble' if model_size_slug == 'ensemble' else model_size_slug
    
    for condition_file in glob.glob(os.path.join(dataset_dir, f'{file_prefix}_*.jsonl')):
        parts = os.path.basename(condition_file).split('_')
        if len(parts) >= 3 and parts[0] == file_prefix:
            if len(parts) >= 3 and parts[2].replace('.jsonl', '').isdigit():
                condition_short_name = parts[1]
            else:
                condition_short_name = parts[1].replace('.jsonl', '')
        else:
            condition_short_name = os.path.basename(condition_file).split('_', 1)[1].replace('.jsonl', '')
        
        with open(condition_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        if data:
            display_name = condition_display_names.get(condition_short_name, condition_short_name)
            metric_fn = data_metric_fn.get(dataset.lower(), calculate_metrics)
            conditions[display_name] = metric_fn(data)
            print(f"Processed {condition_file} -> {display_name}")
    if not conditions:
        print(f"No valid data found in {dataset_dir}")
        return None
    comparison_data = {'conditions': conditions}
    output_file = os.path.join(dataset_dir, f'{dataset}_comparison_{model_size_slug}_0.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Generated and saved new comparison file: {output_file}")
    return comparison_data

def create_radar_plot(data: pd.DataFrame, output_path: str, title: str):
    categories = data.index.tolist()
    series_names = data.columns.tolist()  # Expect conditions here

    non_category_keys = {'total', 'correct', 'total_questions', 'total_correct', 'overall_accuracy', 'total_score'}
    filtered_categories = [cat for cat in categories if isinstance(cat, str) and cat.lower() not in non_category_keys]

    if not filtered_categories:
        print(f"Warning: No valid categories found for radar plot. Available categories: {categories}")
        return
    data = data.loc[filtered_categories]
    categories = filtered_categories

    # Normalize values to float
    for col in data.columns:
        for idx in data.index:
            if isinstance(data.loc[idx, col], dict):
                value = data.loc[idx, col].get('accuracy', list(data.loc[idx, col].values())[0])
                data.loc[idx, col] = float(value)

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 28
    fig, ax = plt.subplots(figsize=(13.2, 13.2), subplot_kw=dict(polar=True))

    sns.set_theme(style="white", font_scale=1.2)
    colorblind_palette = sns.color_palette("colorblind")

    # Consistent condition order and colors
    condition_order = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
    color_mapping = {
        'Normal': colorblind_palette[1],
        'Image Shuffle': colorblind_palette[0],
        'Text Shuffle': colorblind_palette[2],
        'Random': colorblind_palette[3],
    }

    # Plot one series per column (expected to be conditions)
    for i, name in enumerate(series_names):
        color = color_mapping.get(name, colorblind_palette[i % len(colorblind_palette)])
        values = data[name].astype(float).values
        values = np.concatenate((values, [values[0]]))
        ax.plot(
            angles, values,
            linewidth=2.2,
            linestyle='-',
            label=name,
            color=color,
            marker=None,
        )
        ax.fill(angles, values, alpha=0.15, color=color)

    # Customize the polar plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_xticklabels(categories, size=26, color='black')
    ax.tick_params(axis='x', pad=30)

    # Radial axis: 0-100 with clean ticks to match publication style
    ax.set_ylim(0, 100)
    radial_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{t}%" for t in radial_ticks], fontsize=22, color='gray')
    ax.set_rlabel_position(135)
    ax.tick_params(axis='y', pad=-34)
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.35)
    ax.xaxis.grid(True, linestyle='-', color='lightgray', alpha=0.35)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(1.0)

    # Legend order and label mapping consistent with radar_plot_accuracy.py
    # handles, labels = ax.get_legend_handles_labels()
    # handle_map = {lab: h for h, lab in zip(handles, labels)}
    # legend_order = ['Normal', 'Random', 'Image Shuffle', 'Text Shuffle']
    # ordered_handles = [handle_map[l] for l in legend_order if l in handle_map]
    # ordered_labels = [l for l in legend_order if l in handle_map]
    # legend_label_map = {'Image Shuffle': 'Text', 'Text Shuffle': 'Image'}
    # display_labels = [legend_label_map.get(l, l) for l in ordered_labels]
    # ax.legend(
    #     handles=ordered_handles,
    #     labels=display_labels,
    #     loc='upper left',
    #     bbox_to_anchor=(0.84, 1.14),
    #     frameon=True,
    #     fancybox=True,
    #     framealpha=0.35,
    #     edgecolor='#6e6e6e',
    #     facecolor='white',
    #     ncol=1,
    #     fontsize=24,
    #     handlelength=2.2,
    #     columnspacing=1.2,
    #     handletextpad=0.6,
    #     borderaxespad=0.0,
    #     borderpad=0.6,
    #     labelspacing=0.4,
    # )

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(right=0.84, bottom=0.12)
    plt.savefig(output_path, format='pdf', dpi=400, bbox_inches='tight', pad_inches=0.15)
    plt.close()

def create_category_condition_bar_plot(data: pd.DataFrame, output_path: str, title: str):
    # Expect index=categories, columns=conditions
    df_long = data.copy()
    df_long = df_long.reset_index().rename(columns={'index': 'Category'})
    df_long = df_long.melt(id_vars='Category', var_name='Condition', value_name='Score')

    # Ensure numeric
    df_long['Score'] = pd.to_numeric(df_long['Score'], errors='coerce').fillna(0.0)

    # Typography and figure setup (publication-quality, consistent with radar_plot_accuracy.py)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 28

    plt.figure(figsize=(max(10, 1.2 * df_long['Category'].nunique()), 6.5))
    sns.set_theme(style='whitegrid')

    # Consistent condition order and colors
    condition_order = ['Normal', 'Random', 'Image Shuffle', 'Text Shuffle']
    colorblind_palette = sns.color_palette('colorblind')
    color_mapping = {
        'Normal': colorblind_palette[1],
        'Image Shuffle': colorblind_palette[0],
        'Text Shuffle': colorblind_palette[2],
        'Random': colorblind_palette[3],
    }
    palette = [color_mapping.get(c, colorblind_palette[i % len(colorblind_palette)]) for i, c in enumerate(condition_order)]

    ax = sns.barplot(
        data=df_long,
        x='Category', y='Score', hue='Condition',
        hue_order=condition_order,
        palette=palette,
    )
    # Explicitly remove seaborn's automatic legend
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    # Remove axis labels and keep x tick labels horizontal (no angle)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=0)
    # Tick label sizes
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=22)

    # Re-label legend entries to match radar_plot_accuracy display mapping
    # handles, labels = ax.get_legend_handles_labels()
    # legend_label_map = {'Image Shuffle': 'Text', 'Text Shuffle': 'Image'}
    # labels = [legend_label_map.get(l, l) for l in labels]
    # ax.legend(handles=handles, labels=labels, title='Condition', fontsize=24)

    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_condition_radar_plot(model_data: Dict[str, dict], output_dir: str, dataset: str):
    condition_order = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
    

    condition_data = {}
    for model_size, model_results in model_data.items():
        condition_data[model_size] = {}
        for condition in condition_order:
            if condition in model_results['conditions']:
                cond_data = model_results['conditions'][condition]
                if isinstance(cond_data, dict) and 'overall_metrics' in cond_data:
                    overall_metrics = cond_data['overall_metrics']
                    if isinstance(overall_metrics, dict):
                        value = overall_metrics.get('overall_accuracy', overall_metrics.get('total_score', 0))
                    else:
                        value = overall_metrics
                elif isinstance(cond_data, (int, float)):
                    value = cond_data
                else:
                    value = 0
                condition_data[model_size][condition] = float(value)
            else:
                condition_data[model_size][condition] = 0.0
    
    if not condition_data:
        print(f"No condition data found for {dataset}")
        return
    
    # Create DataFrame for plotting
    df = pd.DataFrame(condition_data)
    
    # Create the radar plot
    categories = df.index.tolist()  # These are the conditions
    model_sizes = df.columns.tolist()
    
    # Compute angle for each condition
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Typography and figure setup (publication-quality)
    plt.rcParams['pdf.fonttype'] = 42  # Embed TrueType fonts
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 28
    fig, ax = plt.subplots(figsize=(13.2, 13.2), subplot_kw=dict(polar=True))

    sns.set_theme(style="white", font_scale=1.2)
    # Consistent, colorblind-safe palette with strong contrast
    colorblind_palette = sns.color_palette("colorblind")
    
    # Plot data for each model size
    for i, size in enumerate(model_sizes):
        color = colorblind_palette[i % len(colorblind_palette)]
        values = df[size].astype(float).values
        values = np.concatenate((values, [values[0]]))
        ax.plot(
            angles, values,
            linewidth=2.2,
            linestyle='-',
            label=size.upper(),
            color=color,
            marker=None,
        )
        ax.fill(angles, values, alpha=0.15, color=color)

    # Customize the polar plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_xticklabels(categories, size=26, color='black')
    ax.tick_params(axis='x', pad=30)

    # Radial axis: always show 0-100% with clean ticks
    ax.set_ylim(0, 100)
    radial_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{t}%" for t in radial_ticks], fontsize=22, color='gray')
    # Place radial tick labels at 135° and inside the ring
    ax.set_rlabel_position(135)
    ax.tick_params(axis='y', pad=-34)
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.35)
    ax.xaxis.grid(True, linestyle='-', color='lightgray', alpha=0.35)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(1.0)

    # Legend at top-right (outside axes)
    # ax.legend(
    #     loc='upper left',
    #     bbox_to_anchor=(0.84, 1.14),
    #     frameon=True,
    #     fancybox=True,
    #     framealpha=0.35,
    #     edgecolor='#6e6e6e',
    #     facecolor='white',
    #     ncol=1,
    #     fontsize=24,
    #     handlelength=2.2,
    #     columnspacing=1.2,
    #     handletextpad=0.6,
    #     borderaxespad=0.0,
    #     borderpad=0.6,
    #     labelspacing=0.4,
    # )

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(right=0.84, bottom=0.12)
    
    output_path = os.path.join(output_dir, f'{dataset}_condition_radar.pdf')
    plt.savefig(output_path, format='pdf', dpi=400, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Created condition radar plot: {output_path}")

def create_model_size_comparison_plots(model_data: Dict[str, dict], output_dir: str, dataset: str):
    os.makedirs(output_dir, exist_ok=True)
    

    conditions = set()
    for model_results in model_data.values():
        conditions.update(model_results['conditions'].keys())
    conditions = list(conditions)
    

    sample_metrics = None
    for model_results in model_data.values():
        for cond in conditions:
            if cond in model_results['conditions']:
                overall_metrics = model_results['conditions'][cond]['overall_metrics']
                if isinstance(overall_metrics, dict):
                    sample_metrics = overall_metrics.keys()
                else:
                    sample_metrics = ['overall_accuracy']
                break
        if sample_metrics:
            break
    if not sample_metrics:
        print(f"No metrics found for dataset {dataset}")
        return
    

    for metric in sample_metrics:
        data = []
        for model_size, model_results in model_data.items():
            for condition in conditions:
                if condition in model_results['conditions']:
                    cond_data = model_results['conditions'][condition]

                    if isinstance(cond_data, dict) and 'overall_metrics' in cond_data:
                        overall_metrics = cond_data['overall_metrics']
                        if isinstance(overall_metrics, dict):
                            value = overall_metrics.get(metric, 0)
                        else:
                            value = overall_metrics
                    elif isinstance(cond_data, (int, float)):
                        value = cond_data
                    else:
                        value = 0

                    if isinstance(value, (int, float)):
                        display_size = 'Ensemble' if model_size.lower() == 'ensemble' else model_size.upper()
                        data.append({
                            'Model Size': display_size,
                            'Condition': condition,
                            'Score': float(value)
                        })

        if not data:
            continue

        df = pd.DataFrame(data)
        # Order conditions consistently and map legend labels
        condition_order = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
        df['Condition'] = pd.Categorical(df['Condition'], categories=condition_order, ordered=True)
        # Order model sizes, if present
        model_order = [label for label in ['8B', '13B', '34B', 'Ensemble'] if label in df['Model Size'].unique()]
        df['Model Size'] = pd.Categorical(df['Model Size'], categories=model_order, ordered=True)
        df = df.sort_values(['Model Size', 'Condition'])

        # Typography and theme
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 28
        sns.set_theme(style='whitegrid')

        plt.figure(figsize=(13.2, 6.8))
        # Consistent condition colors
        colorblind_palette = sns.color_palette('colorblind')
        color_mapping = {
            'Normal': colorblind_palette[1],
            'Image Shuffle': colorblind_palette[0],
            'Text Shuffle': colorblind_palette[2],
            'Random': colorblind_palette[3],
        }
        palette = [color_mapping[c] for c in condition_order]

        ax = sns.barplot(
            data=df, x='Model Size', y='Score', hue='Condition',
            hue_order=condition_order, palette=palette
        )
        # Explicitly remove seaborn's automatic legend
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        # Remove axis labels and avoid rotation
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.xticks(rotation=0)
        ax.tick_params(axis='x', labelsize=26)
        ax.tick_params(axis='y', labelsize=22)

        # Legend 2x2 and relabel entries
        # handles, labels = ax.get_legend_handles_labels()
        # legend_label_map = {'Image Shuffle': 'Text', 'Text Shuffle': 'Image'}
        # labels = [legend_label_map.get(l, l) for l in labels]
        # ax.legend(handles=handles, labels=labels, loc='upper center', ncol=2, fontsize=24, frameon=True)

        # Bar labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=14)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{dataset}_{metric}_model_size_comparison.pdf')
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    

    # Skip creating the condition-level radar plot; focus on category vs condition plots instead
    

    # Build per-model-size category vs condition plots
    condition_display_order = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
    for model_size, model_results in model_data.items():
        # Aggregate all categories across conditions for this model_size
        condition_to_categories = {}
        all_categories = set()
        for condition in condition_display_order:
            if condition not in model_results['conditions']:
                continue
            cond_data = model_results['conditions'][condition]
            if isinstance(cond_data, dict) and 'category_scores' in cond_data:
                category_scores = cond_data['category_scores']
                filtered_scores = {}
                for cat, score in category_scores.items():
                    if isinstance(score, dict):
                        if 'score' in score:
                            filtered_scores[cat] = score['score']
                        elif 'accuracy' in score:
                            filtered_scores[cat] = score['accuracy']
                    else:
                        filtered_scores[cat] = score
                non_category_keys = {'total', 'correct', 'total_questions', 'total_correct', 'overall_accuracy', 'total_score'}
                meaningful = {k: float(v) for k, v in filtered_scores.items() if isinstance(k, str) and k.lower() not in non_category_keys}
                if meaningful:
                    condition_to_categories[condition] = meaningful
                    all_categories.update(meaningful.keys())

        if not condition_to_categories:
            continue

        # Create DataFrame with index=categories, columns=conditions
        all_categories = sorted(list(all_categories))
        df = pd.DataFrame(index=all_categories, columns=[c for c in condition_display_order if c in condition_to_categories])
        for cond, cat_scores in condition_to_categories.items():
            for cat in all_categories:
                df.loc[cat, cond] = float(cat_scores.get(cat, 0.0))

        # Decide plot type: radar if more than 4 categories, else grouped bar
        num_categories = len(all_categories)
        if num_categories > 4:
            output_path = os.path.join(output_dir, f'{dataset}_{model_size}_category_condition_radar.pdf')
            create_radar_plot(df, output_path, f'{dataset.upper()} Category Performance - {model_size.upper()}')
        else:
            output_path = os.path.join(output_dir, f'{dataset}_{model_size}_category_condition_bars.pdf')
            create_category_condition_bar_plot(df, output_path, f'{dataset.upper()} Category Performance - {model_size.upper()}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='.', help='Base directory containing answer directories (default: current directory)')
    parser.add_argument('--datasets', type=str, nargs='+', help='Specific datasets to process (default: auto-detect all available)')
    parser.add_argument('--output_dir', type=str, help='Output directory for plots (default: base_dir/model_size_comparison_plots)')
    parser.add_argument('--force', action='store_true', help='Force reprocess a dataset even if outputs already exist')
    args = parser.parse_args()
    

    model_sizes = ['8b', '13b', '34b']
    

    if not args.datasets:
        args.datasets = detect_available_datasets(args.base_dir, model_sizes)
        print(f"Auto-detected datasets: {args.datasets}")
    
    if not args.datasets:
        print("No datasets found. Please check the base directory and model sizes.")
        return
    

    if not args.output_dir:
        args.output_dir = os.path.join(args.base_dir, 'model_size_comparison_plots')
    
    for dataset in args.datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        dataset_output_dir = os.path.join(args.output_dir, dataset)

        # Skip datasets that appear already processed unless forced
        if (not args.force
            and os.path.isdir(dataset_output_dir)
            and any(name.endswith('.pdf') for name in os.listdir(dataset_output_dir))):
            print(f"Skipping dataset {dataset} (outputs already exist). Use --force to regenerate.")
            continue

        # Create ensemble predictions; if this errors, skip dataset
        try:
            print(f"Creating ensemble predictions for {dataset}...")
            ensemble_dir = create_ensemble_predictions(model_sizes, args.base_dir, dataset)
            if ensemble_dir:
                print(f"Ensemble predictions created in: {ensemble_dir}")
        except Exception as e:
            print(f"Error creating ensemble predictions for {dataset}: {e}. Skipping dataset.")
            continue

        model_data = {}
        # Load/generate comparison data for each model size; on error, skip that size
        for size in model_sizes:
            answers_dir = os.path.join(args.base_dir, f'answers_{size}')
            if not os.path.exists(answers_dir):
                print(f"Warning: Answers directory {answers_dir} does not exist")
                continue
            try:
                comparison_pattern = os.path.join(answers_dir, dataset, f'{dataset}_comparison_*.json')
                comparison_files = glob.glob(comparison_pattern)
                if comparison_files:
                    comparison_file = comparison_files[0]
                    with open(comparison_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    model_data[size] = data
                else:
                    print(f"Generating comparison data for {size}...")
                    comparison_data = load_comparison_data(answers_dir, dataset)
                    if comparison_data:
                        model_data[size] = comparison_data
            except Exception as e:
                print(f"Error preparing comparison data for size {size} on {dataset}: {e}. Skipping this size.")
                continue

        # Also include ensemble predictions in model_data if available; on error, ignore ensemble
        ensemble_answers_dir = os.path.join(args.base_dir, 'answers_ensemble')
        if os.path.exists(ensemble_answers_dir):
            try:
                comparison_pattern = os.path.join(ensemble_answers_dir, dataset, f'{dataset}_comparison_*.json')
                comparison_files = glob.glob(comparison_pattern)
                if comparison_files:
                    with open(comparison_files[0], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    model_data['ensemble'] = data
                else:
                    comparison_data = load_comparison_data(ensemble_answers_dir, dataset)
                    if comparison_data:
                        model_data['ensemble'] = comparison_data
            except Exception as e:
                print(f"Error preparing comparison data for ensemble on {dataset}: {e}. Skipping ensemble for this dataset.")

        if not model_data:
            print(f"No comparison data found for {dataset}; skipping.")
            continue

        try:
            create_model_size_comparison_plots(model_data, dataset_output_dir, dataset)
            print(f"Created comparison plots in: {dataset_output_dir}")
        except Exception as e:
            print(f"Error creating plots for {dataset}: {e}. Skipping dataset.")



if __name__ == "__main__":
    main() 