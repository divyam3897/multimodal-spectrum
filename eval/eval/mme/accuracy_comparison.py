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
    # Try to find comparison JSON files for different datasets
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
    
    category_rows = [{'condition': cond, 'category': cat, **scores} for cond, res in data['conditions'].items() for cat, scores in res['category_scores'].items()]
    category_df = pd.DataFrame(category_rows)

    s_metrics_ratio_df = None
    s_metrics_prop_df = None
    if data.get('s_metrics') and data['s_metrics']:
        if 'ratio' in data['s_metrics'] and data['s_metrics']['ratio']:
            s_metrics_ratio_df = pd.DataFrame.from_dict(data['s_metrics']['ratio'], orient='index').rename(columns={'S_rel': '$S_{\mathrm{rel}}$', 'S_image': '$S_{\mathrm{image}}$', 'S_text': '$S_{\mathrm{text}}$'})
        if 'proportional' in data['s_metrics'] and data['s_metrics']['proportional']:
            s_metrics_prop_df = pd.DataFrame.from_dict(data['s_metrics']['proportional'], orient='index').rename(columns={'S_rel': '$S_{\mathrm{rel}}$', 'S_image': '$S_{\mathrm{image}}$', 'S_text': '$S_{\mathrm{text}}$'})

    print(f"Successfully processed {dataset_name} JSON data into DataFrames.")
    return {
        'overall': overall_df,
        'category_details': category_df,
        's_metrics_ratio': s_metrics_ratio_df,
        's_metrics_proportional': s_metrics_prop_df,
        'dataset_name': dataset_name
    }

def create_publication_bar_chart(plot_df, output_dir, file_prefix, title):
    print(f"  -> Generating Bar Chart: {file_prefix}_bar.pdf")
    sns.set_theme(style="ticks", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = sns.color_palette("viridis", n_colors=len(plot_df.columns))
    plot_df.plot(kind='bar', ax=ax, width=0.75, color=colors, zorder=2)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)

    # ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontsize=14)
    # ax.set_xlabel(None)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    
    min_val, max_val = plot_df.min().min(), plot_df.max().max()
    padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
    ax.set_ylim((min_val - padding), max_val + padding)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), frameon=True, fancybox=True, shadow=True, ncol=len(plot_df.columns), fontsize=14, title='Metric')
    fig.subplots_adjust(bottom=0.3)
    output_file = os.path.join(output_dir, f"{file_prefix}_bar.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

def create_publication_radar_chart(plot_df, output_dir, file_prefix, title):
    print(f"  -> Generating Radar Chart: {file_prefix}_radar.pdf")
    labels = plot_df.index.values
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    sns.set_theme(style="white", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    colors = sns.color_palette("colorblind", n_colors=len(plot_df.columns))

    for i, (condition) in enumerate(plot_df.columns):
        values = plot_df[condition].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2.5, label=condition)
        ax.fill(angles, values, color=colors[i], alpha=0.2)

    max_val = plot_df.max().max()
    ax.set_ylim(bottom=0, top=max_val * 1.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=18)

    # ax.set_title(title, size=20, y=1.1, fontweight='bold')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), frameon=True, fancybox=True, shadow=True, ncol=len(plot_df.columns), fontsize=20)
    output_file = os.path.join(output_dir, f"{file_prefix}_radar.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

def create_dot_plot(plot_df, output_dir, file_prefix, title):
    print(f"  -> Generating Dot Plot: {file_prefix}_dot.pdf")
    plot_df_melted = plot_df.reset_index().rename(columns={'index': 'category'}).melt(id_vars='category', var_name='Metric', value_name='Value')

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.stripplot(data=plot_df_melted, x='Value', y='category', hue='Metric', palette='colorblind', dodge=True, s=8, ax=ax)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='y')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True, fancybox=True, shadow=True, ncol=len(plot_df.columns), fontsize=14, title='Metric')
    fig.subplots_adjust(bottom=0.2)
    output_file = os.path.join(output_dir, f"{file_prefix}_dot.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

def generate_all_plots(dataframes, output_dir, dataset_name="MME"):
    generated_files = []
    
    # --- 1. Category-Level Accuracy and Score Plots ---
    category_df = dataframes['category_details']
    
    # Define metrics based on dataset type
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
        print(f"\n=== GENERATING PLOTS FOR: {title_part} ===")
        pivot_data = category_df.pivot_table(index='category', columns='condition', values=metric_col)
        title = f'{dataset_name} {title_part} Comparison'
        file_prefix = f'category_{file_prefix_part}_{dataset_name.lower()}'
        
        generated_files.append(create_publication_bar_chart(pivot_data, output_dir, file_prefix, title))
        generated_files.append(create_publication_radar_chart(pivot_data, output_dir, file_prefix, title))

    # --- 2. S-Metrics (Ratio) Plots ---
    if dataframes.get('s_metrics_ratio') is not None:
        print("\n=== GENERATING S-METRICS PLOTS (PERFORMANCE GAP RATIO) ===")
        s_metrics_data = dataframes['s_metrics_ratio']
        title = f'{dataset_name} S-Metrics (Performance Gap Ratio)'
        file_prefix = f's_metrics_ratio_{dataset_name.lower()}'
        generated_files.append(create_dot_plot(s_metrics_data, output_dir, file_prefix, title))
    
    # --- 3. S-Metrics (Proportional) Plots ---
    if dataframes.get('s_metrics_proportional') is not None:
        print("\n=== GENERATING S-METRICS PLOTS (PROPORTIONAL CONTRIBUTION) ===")
        s_metrics_data = dataframes['s_metrics_proportional']
        title = f'{dataset_name} S-Metrics (Proportional Contribution)'
        file_prefix = f's_metrics_proportional_{dataset_name.lower()}'
        generated_files.append(create_dot_plot(s_metrics_data, output_dir, file_prefix, title))

    # --- 4. Overall Metrics Plots ---
    print("\n=== GENERATING OVERALL COMPARISON PLOTS ===")
    overall_df = dataframes['overall']
    if not overall_df.empty:
        # Define overall metrics based on dataset type
        if dataset_name == "MME":
            overall_metrics_to_plot = [
                ('overall_accuracy', 'Overall Accuracy'),
                ('overall_accuracy_plus', 'Overall Accuracy+'),
                ('total_score', 'Total Score'),
                ('perception_score', 'Perception Score'),
                ('cognition_score', 'Cognition Score')
            ]
        else:  # MMMU or MMMU-Pro
            overall_metrics_to_plot = [
                ('overall_accuracy', 'Overall Accuracy'),
                ('total_score', 'Total Score'),
            ]
        
        for metric_col, metric_label in overall_metrics_to_plot:
            if metric_col in overall_df.columns:
                print(f"  -> Generating Bar Chart: overall_{metric_col}.pdf")
                sns.set_theme(style="ticks", font_scale=1.2)
                fig, ax = plt.subplots(figsize=(10, 7))
                values = overall_df[metric_col]
                ax.bar(overall_df.index, values, color=sns.color_palette("viridis", n_colors=len(overall_df)))
                ax.bar_label(ax.containers[0], fmt='%.3f', fontweight='bold', padding=3)
                # ax.set_title(f'{dataset_name} {metric_label} by Condition', fontsize=16, fontweight='bold', pad=20)
                ax.set_ylabel(metric_label, fontsize=14)
                # ax.set_xlabel('Experimental Condition', fontsize=14)
                ax.yaxis.grid(True, linestyle='--', alpha=0.5)
                ax.spines[['top', 'right']].set_visible(False)
                max_val = values.max()
                ax.set_ylim(0, max_val * 1.15)
                plt.tight_layout()
                output_file = os.path.join(output_dir, f'overall_{metric_col}_{dataset_name.lower()}.pdf')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(output_file)
    
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
    
    # Use auto-detected dataset name if not specified
    dataset_name = args.dataset_name if args.dataset_name else dataframes.get('dataset_name', "Unknown")
        
    generated_files = generate_all_plots(dataframes, output_dir, dataset_name)


if __name__ == "__main__":
    main()