import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def load_accuracy_data(dataset_dirs):
    """
    Load accuracy data from JSON files for multiple datasets.
    Each JSON is expected to have overall_metrics for different conditions.
    """
    condition_mapping = {
        'Normal': 'nrm',
        'Image Shuffle': 'img',
        'Text Shuffle': 'txt',
        'Random': 'rdm'
    }
    accuracy_data = {}
    datasets_less_than_1 = set()
    
    for condition in condition_mapping.keys():
        accuracy_data[condition] = {}
    
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        json_pattern = os.path.join(dataset_dir, f'{dataset_name}_comparison_*.json')
        json_files = glob.glob(json_pattern)
        if json_files:
            # Sort files by name to ignore date/time stamp, taking the last one alphabetically if needed
            latest_json_file = sorted(json_files)[-1]
            with open(latest_json_file, 'r') as f:
                data = json.load(f)
                for condition in condition_mapping.keys():
                    if condition in data['conditions']:
                        metrics = data['conditions'][condition].get('overall_metrics', 0.0)
                        if isinstance(metrics, dict):
                            # Extract overall_accuracy if available, otherwise use total_score or default to 0.0
                            value = metrics.get('overall_accuracy', metrics.get('total_score', 0.0))
                            accuracy_data[condition][dataset_name] = value
                            if 0 < value < 1:
                                datasets_less_than_1.add(dataset_name)
                        else:
                            accuracy_data[condition][dataset_name] = metrics
                            if 0 < metrics < 1:
                                datasets_less_than_1.add(dataset_name)
                    else:
                        accuracy_data[condition][dataset_name] = 0.0
                        print(f"Warning: {condition} not found in {latest_json_file}, setting accuracy to 0.0")
        else:
            for condition in condition_mapping.keys():
                accuracy_data[condition][dataset_name] = 0.0
            print(f"Warning: No comparison JSON found for {dataset_name}, setting all accuracies to 0.0")
    
    return accuracy_data

def plot_radar_chart(accuracy_data, model_slug):
    """
    Generate a radar chart for accuracy across different datasets for multiple conditions.
    """
    datasets = list(next(iter(accuracy_data.values())).keys())
    print(f"Datasets for plotting: {datasets}")
    if not all(isinstance(d, str) for d in datasets):
        raise ValueError(f"Non-string dataset names found: {datasets}")
    num_vars = len(datasets)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
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
    condition_order = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
    color_mapping = {
        'Normal': colorblind_palette[1],        # Blue
        'Image Shuffle': colorblind_palette[0], # Orange
        'Text Shuffle': colorblind_palette[2],  # Green
        'Random': colorblind_palette[3],        # Red
    }
    # Use subtle, consistent markers for a cleaner look
    marker_mapping = {
        'Normal': None,
        'Image Shuffle': None,
        'Text Shuffle': None,
        'Random': None,
    }

    # Draw each condition
    for condition in condition_order:
        if condition not in accuracy_data:
            continue
        accuracies = accuracy_data[condition]
        values = [float(accuracies[dataset]) for dataset in datasets]
        if not all(isinstance(v, (int, float)) for v in values):
            raise ValueError(f"Non-numeric values found for {condition}: {values}")
        values += values[:1]  # close loop
        color = color_mapping[condition]
        marker = marker_mapping[condition]
        ax.plot(
            angles, values,
            linewidth=2.2,
            linestyle='-',
            label=condition,
            color=color,
            marker=None,
        )
        ax.fill(angles, values, alpha=0.15, color=color)

    # Customize the polar plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Remap some dataset labels for display
    display_map = {
        'mmbench_en': 'mmb_en',
        'mmbench_cn': 'mmb_cn',
    }
    display_datasets = [display_map.get(d, d) for d in datasets]
    ax.set_thetagrids(np.degrees(angles[:-1]), display_datasets)
    ax.set_xticklabels(display_datasets, size=26, color='black')
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
    handles, labels = ax.get_legend_handles_labels()
    handle_map = {lab: h for h, lab in zip(handles, labels)}
    legend_order = ['Normal', 'Random', 'Image Shuffle', 'Text Shuffle']
    ordered_handles = [handle_map[l] for l in legend_order if l in handle_map]
    ordered_labels = [l for l in legend_order if l in handle_map]
    # Swap display names: show Image Shuffle as "Text Shuffle" and Text Shuffle as "Image Shuffle"
    legend_label_map = {'Image Shuffle': 'Text', 'Text Shuffle': 'Image'}
    display_labels = [legend_label_map.get(l, l) for l in ordered_labels]
    ax.legend(
        handles=ordered_handles,
        labels=display_labels,
        loc='upper left',
        bbox_to_anchor=(0.84, 1.14),
        frameon=True,
        fancybox=True,
        framealpha=0.35,
        edgecolor='#6e6e6e',
        facecolor='white',
        ncol=1,
        fontsize=24,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
        borderpad=0.6,
        labelspacing=0.4,
    )

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(right=0.84, bottom=0.12)
    output_file_pdf = f'./radar_plot_accuracy_{model_slug}.pdf'
    plt.savefig(output_file_pdf, format='pdf', dpi=400, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Radar plot saved to {output_file_pdf}")

    # Print accuracy values grouped by dataset
    print("\nAccuracy values by dataset (Normal, Image Shuffle, Text Shuffle, Random):")
    conditions = ['Normal', 'Image Shuffle', 'Text Shuffle', 'Random']
    for dataset in datasets:
        values = [accuracy_data[cond][dataset] for cond in conditions]
        formatted_values = [f"{v:.2f}" for v in values]
        print(f"{dataset}: {dict(zip(conditions, formatted_values))}")

def main():
    # List of directories containing the dataset results
    # Adjust these paths based on your actual directory structure
    model_slug = 'answers_ensemble'
    dataset_dirs = [
        f'./{model_slug}/mme',
        f'./{model_slug}/pope',
        f'./{model_slug}/coco',
        
        f'./{model_slug}/realworldqa',

        f'./{model_slug}/gqa',
        f'./{model_slug}/omni',
        f'./{model_slug}/vstar',
        f'./{model_slug}/blink', 
        f'./{model_slug}/scienceqa',
        f'./{model_slug}/mmvp',
        f'./{model_slug}/mmbench_en',
        f'./{model_slug}/mmbench_cn',
        f'./{model_slug}/seed', 
        f'./{model_slug}/vizwiz',
        f'./{model_slug}/qbench',
        f'./{model_slug}/chartqa',
        f'./{model_slug}/textvqa',
        f'./{model_slug}/ai2d',
        f'./{model_slug}/ocrbench',
        f'./{model_slug}/mmstar',
        f'./{model_slug}/mmmu',
        f'./{model_slug}/mmmu',
        f'./{model_slug}/mmmupro',
        # './answers_8b/ade',
        # './answers_8b/docvqa',
        # './answers_8b/infovqa',
        # './answers_8b/mmvet',
        # './answers_8b/stvqa',
        # f'./{model_slug}/synthdog',

        f'./{model_slug}/mathvista'
    ]
    
    accuracy_data = load_accuracy_data(dataset_dirs)
    # print(accuracy_data)
    plot_radar_chart(accuracy_data, model_slug)

if __name__ == '__main__':
    main() 