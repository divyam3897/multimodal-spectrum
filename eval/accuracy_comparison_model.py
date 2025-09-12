import json
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import seaborn as sns
from typing import Union

def load_and_process_data(answers_dir: str) -> Union[dict, None]:
    dataset_patterns = [
        ("mme_comparison_*.json", "MME"),
        ("mmmu_comparison_*.json", "MMMU"),
        ("mmmupro_comparison_*.json", "MMMU-Pro")
    ]
    
    json_files = []
    dataset_name = None
    
    for pattern, name in dataset_patterns:
        json_pattern = os.path.join(answers_dir, pattern)
        found_files = glob.glob(json_pattern)
        if found_files:
            json_files = found_files
            dataset_name = name
            break
    
    if not json_files:
        print(f"Error: No comparison JSON file found in '{answers_dir}'.")
        print("Looking for: mme_comparison_*.json, mmmu_comparison_*.json, or mmmupro_comparison_*.json")
        return None
        
    latest_json_file = max(json_files, key=os.path.getmtime)
    print(f"Loading {dataset_name} data from: {os.path.basename(latest_json_file)}")

    with open(latest_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    overall_df = pd.DataFrame.from_dict({cond: res['overall_metrics'] for cond, res in data['conditions'].items()}, orient='index')
    
    category_rows = []
    for cond, res in data['conditions'].items():
        if 'category_scores' in res:
            for cat, scores in res['category_scores'].items():
                row = {'condition': cond, 'category': cat}
                if isinstance(scores, dict):
                    row.update(scores)
                elif isinstance(scores, (int, float)):
                    # Handle simplified format where score is a single float
                    row['accuracy'] = scores
                    row['score'] = scores
                category_rows.append(row)

    category_df = pd.DataFrame(category_rows)

    print(f"Successfully processed {dataset_name} JSON data into DataFrames.")
    print(f"Found conditions for {dataset_name}: {overall_df.index.tolist()}")
    
    print(f"Renamed conditions for plotting to: {overall_df.index.tolist()}")

    return {
        'overall': overall_df,
        'category_details': category_df,
        'dataset_name': dataset_name
    }

def order_conditions(df, axis=0):
    condition_order = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
    
    if axis == 0: 
        available_conditions = [cond for cond in condition_order if cond in df.index]
        return df.reindex(available_conditions)
    else:  
        available_conditions = [cond for cond in condition_order if cond in df.columns]
        return df[available_conditions]

def create_publication_radar_chart(plot_df, output_dir, file_prefix, title=None):
    labels = plot_df.index.values
    num_vars = len(labels)
    plot_df = order_conditions(plot_df, axis=1)

    # Calculate angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    sns.set_theme(style="white", font_scale=1.2)
    colors = sns.color_palette("colorblind", n_colors=len(plot_df.columns))
    colorblind_palette = sns.color_palette("colorblind")
    color_mapping = {
        'Normal': colorblind_palette[1],        # Blue
        'Text Shuffle': colorblind_palette[2],  # Green
        'Random': colorblind_palette[3],        # Red
        'Image Shuffle': colorblind_palette[0], # Orange
    }
    for i, condition in enumerate(plot_df.columns):
        values = plot_df[condition].values.flatten().tolist()
        values += values[:1] # Complete the loop
        color = color_mapping.get(condition)
        ax.plot(angles, values, color=color, linewidth=1.8, label=condition)
        ax.fill(angles, values, color=color, alpha=0.12)
    max_val = plot_df.values.max()
    ax.set_ylim(0, max_val * 1.1)

    ax.grid(color='gray', linestyle='--',  alpha=0.5)

    ax.spines['polar'].set_color('black')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=20, color='black')
    ax.tick_params(axis='x', pad=15) 

    ax.tick_params(axis='y', labelsize=16, color='gray')

    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1),
              frameon=False, 
              ncol=1, fontsize=22)

    output_file = os.path.join(output_dir, f"{file_prefix}_radar.pdf")
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  -> Successfully saved to {output_file}")
    
    return output_file




def create_overall_bar_chart(values, labels, output_dir, file_prefix, metric_label, dataset_name):
    print(f"  -> Generating Bar Chart: {file_prefix}.pdf")
    
    condition_order = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
    ordered_labels = []
    ordered_values = []
    
    for condition in condition_order:
        if condition in labels:
            idx = list(labels).index(condition)
            ordered_labels.append(condition)
            ordered_values.append(values[idx])
    
    sns.set_theme(style="ticks", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", n_colors=len(ordered_values))
    bars = ax.bar(ordered_labels, ordered_values, color=colors)
    
    for bar, value in zip(bars, ordered_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel(metric_label, fontsize=14, weight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    max_val = max(ordered_values)
    ax.set_ylim(0, max_val * 1.15)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{file_prefix}.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

def generate_all_plots(dataframes, output_dir, dataset_name="MME"):
    generated_files = []
    
    category_df = dataframes['category_details']
    
    if dataset_name == "MME":
        metrics_to_plot = [
            ('accuracy', 'Raw Accuracy', 'accuracy'),
            ('accuracy_plus', 'Accuracy Plus Score', 'acc_plus'),
            ('score', 'MME Combined Score', 'mme_score'),
        ]
    else:  # MMMU or MMMU-Pro
        metrics_to_plot = [
            ('accuracy', 'Raw Accuracy', 'accuracy'),
            ('score', f'{dataset_name} Score', 'score'),
        ]

    for metric_col, title_part, file_prefix_part in metrics_to_plot:
        if metric_col not in category_df.columns:
            print(f"Skipping '{title_part}' radar plot: metric not found in data.")
            continue
        
        print(f"\n=== GENERATING RADAR PLOT FOR: {title_part} ===")
        pivot_data = category_df.pivot_table(index='category', columns='condition', values=metric_col)
        title = f'{dataset_name} {title_part} Comparison'
        file_prefix = f'category_{file_prefix_part}_{dataset_name.lower()}'
        
        generated_files.append(create_publication_radar_chart(pivot_data, output_dir, file_prefix, title))

    print("\n=== GENERATING OVERALL COMPARISON PLOTS ===")
    overall_df = dataframes['overall']
    if not overall_df.empty:
        if dataset_name == "MME":
            overall_metrics_to_plot = [
                ('overall_accuracy', 'Overall Accuracy'),
                ('overall_accuracy_plus', 'Overall Accuracy+'),
                ('total_score', 'Total Score'),
                ('perception_score', 'Perception Score'),
                ('cognition_score', 'Cognition Score')
            ]
        else:  
            overall_metrics_to_plot = [
                ('overall_accuracy', 'Overall Accuracy'),
                ('total_score', 'Total Score'),
            ]
        
        for metric_col, metric_label in overall_metrics_to_plot:
            if metric_col in overall_df.columns:
                values = overall_df[metric_col].values
                labels = overall_df.index.tolist()
                file_prefix = f'overall_{metric_col}_{dataset_name.lower()}'
                generated_files.append(create_overall_bar_chart(values, labels, output_dir, file_prefix, metric_label, dataset_name))
    
    return generated_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset_name', type=str, default=None, help="Dataset name (auto-detected if not specified)")
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.output_dir else args.answers_dir
    os.makedirs(output_dir, exist_ok=True)
    
    dataframes = load_and_process_data(args.answers_dir)
    if not dataframes:
        return
    
    dataset_name = args.dataset_name if args.dataset_name else dataframes.get('dataset_name', "Unknown")
        
    generated_files = generate_all_plots(dataframes, output_dir, dataset_name)

if __name__ == "__main__":
    main()