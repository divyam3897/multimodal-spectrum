import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

CONDITION_MAPPING = {
    'Normal': 'nrm',
    'Image Shuffle': 'img',
    'Text Shuffle': 'txt',
    'Random': 'rdm'
}


def load_accuracy_data(dataset_dirs):
    accuracy_data = {}
    for condition in CONDITION_MAPPING:
        accuracy_data[condition] = {}
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        json_pattern = os.path.join(dataset_dir, f'{dataset_name}_comparison_*.json')
        json_files = glob.glob(json_pattern)
        if json_files:
            latest_json_file = sorted(json_files)[-1]
            with open(latest_json_file, 'r') as f:
                data = json.load(f)
                for condition in CONDITION_MAPPING:
                    value = 0.0
                    if condition in data['conditions']:
                        cond_data = data['conditions'][condition]
                        metrics = cond_data['overall_metrics'] if 'overall_metrics' in cond_data else 0.0
                        if isinstance(metrics, dict):
                            value = metrics['overall_accuracy'] if 'overall_accuracy' in metrics else (metrics['total_score'] if 'total_score' in metrics else 0.0)
                        else:
                            value = metrics
                    accuracy_data[condition][dataset_name] = value
        else:
            for condition in CONDITION_MAPPING:
                accuracy_data[condition][dataset_name] = 0.0
    return accuracy_data


def plot_radar_chart(accuracy_data, model_slug, human_scores, group1, group2, output_suffix=""):
    datasets = list(next(iter(accuracy_data.values())).keys())
    if not all(isinstance(d, str) for d in datasets):
        raise ValueError(f"Non-string dataset names found: {datasets}")
    num_vars = len(datasets)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 28
    fig, ax = plt.subplots(figsize=(13.2, 13.2), subplot_kw=dict(polar=True))
    sns.set_theme(style="white", font_scale=1.2)
    colorblind_palette = sns.color_palette("colorblind")

    accuracy_data = dict(accuracy_data)
    accuracy_data['Human'] = {
        d: (float(human_scores[d]) if d in human_scores and human_scores[d] is not None else np.nan)
        for d in list(next(iter(accuracy_data.values())).keys())
    }
    _datasets_list = list(next(iter(accuracy_data.values())).keys())
    for _d in _datasets_list:
        _n = float(accuracy_data['Normal'][_d]) if _d in accuracy_data['Normal'] else 0.0
        _i = float(accuracy_data['Image Shuffle'][_d]) if _d in accuracy_data['Image Shuffle'] else 0.0
        _t = float(accuracy_data['Text Shuffle'][_d]) if _d in accuracy_data['Text Shuffle'] else 0.0
        _r = float(accuracy_data['Random'][_d]) if _d in accuracy_data['Random'] else 0.0
        if _n == 0.0 and _i == 0.0 and _t == 0.0 and _r == 0.0:
            accuracy_data['Human'][_d] = np.nan

    condition_order = ['Normal', 'Random', 'Image Shuffle', 'Text Shuffle', 'Human']
    color_mapping = {
        'Normal': colorblind_palette[1],
        'Image Shuffle': colorblind_palette[0],
        'Text Shuffle': colorblind_palette[2],
        'Random': colorblind_palette[3],
        'Human': '#8B4513',
    }
    marker_mapping = {'Normal': None, 'Image Shuffle': None, 'Text Shuffle': None, 'Random': None, 'Human': 'o'}

    for condition in condition_order:
        if condition not in accuracy_data:
            continue
        accuracies = accuracy_data[condition]
        values = [float(accuracies[dataset]) for dataset in datasets]
        values += values[:1]
        color = color_mapping[condition]
        marker = marker_mapping[condition]
        if condition == 'Human':
            ax.plot(angles, values, linewidth=3.0, linestyle='--', label=condition, color=color, marker=marker)
        else:
            ax.plot(angles, values, linewidth=2.2, linestyle='-', label=condition, color=color, marker=None)
            ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    display_map = {'mmbench_en': 'mmb_en', 'mmbench_cn': 'mmb_cn'}
    display_datasets = [display_map[d] if d in display_map else d for d in datasets]
    ax.set_thetagrids(np.degrees(angles[:-1]), display_datasets)
    ax.set_xticklabels(display_datasets, size=26, color='black')
    ax.tick_params(axis='x', pad=30)

    group1_color = '#8B008B'
    group2_color = '#FF6347'
    for idx, tick_label in enumerate(ax.get_xticklabels()):
        dataset_key = datasets[idx]
        if dataset_key in group1:
            tick_label.set_color(group1_color)
            tick_label.set_fontweight('semibold')
        elif dataset_key in group2:
            tick_label.set_color(group2_color)
            tick_label.set_fontweight('semibold')

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

    handles, labels = ax.get_legend_handles_labels()
    handle_map = {lab: h for h, lab in zip(handles, labels)}
    legend_order = ['Normal', 'Random', 'Image Shuffle', 'Text Shuffle']
    ordered_handles = [handle_map[l] for l in legend_order if l in handle_map]
    ordered_labels = [l for l in legend_order if l in handle_map]
    legend_label_map = {'Image Shuffle': 'Text', 'Text Shuffle': 'Image'}
    display_labels = [legend_label_map[l] if l in legend_label_map else l for l in ordered_labels]
    ax.legend(
        handles=ordered_handles,
        labels=display_labels,
        loc='upper left',
        bbox_to_anchor=(0.60, 1.12),
        frameon=True,
        fancybox=True,
        framealpha=0.35,
        edgecolor='#6e6e6e',
        facecolor='white',
        ncol=2,
        fontsize=24,
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.6,
        borderaxespad=0.0,
        borderpad=0.6,
        labelspacing=0.3,
    )

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(right=0.96, bottom=0.12)
    out_name = f'radar_plot_accuracy_{model_slug}{output_suffix}.pdf'
    plt.savefig(out_name, format='pdf', dpi=400, bbox_inches='tight', pad_inches=0.15)
    plt.close()

def main():
    model_slug = 'answers_ensemble'

    config_g1 = {
        'dataset_dirs': [
            f'./{model_slug}/vizwiz',
            f'./{model_slug}/gqa',
            f'./{model_slug}/mme',
            f'./{model_slug}/seed',
            f'./{model_slug}/mmbench_en',
            f'./{model_slug}/mmbench_cn',
            f'./{model_slug}/scienceqa',
            f'./{model_slug}/mmmu',
            f'./{model_slug}/mmmupro',
            f'./{model_slug}/mathvista',
        ],
        'human_scores': {
            'scienceqa': 88.4, 'mmmu': 88.6, 'mmmupro': 85.6, 'gqa': 89.3,
            'vizwiz': 75.0, 'mathvista': 60.0,
        },
        'group1': {'vizwiz', 'gqa', 'mme', 'mmbench_en', 'mmbench_cn', 'seed'},
        'group2': {'scienceqa', 'mmmu', 'mmmupro', 'mathvista'},
        'output_suffix': '_g1',
    }

    config_g2 = {
        'dataset_dirs': [
            f'./{model_slug}/coco',
            f'./{model_slug}/pope',
            f'./{model_slug}/ade',
            f'./{model_slug}/mmstar',
            f'./{model_slug}/qbench',
            f'./{model_slug}/mmvp',
            f'./{model_slug}/vstar',
            f'./{model_slug}/blink',
            f'./{model_slug}/realworldqa',
            f'./{model_slug}/omni',
            f'./{model_slug}/ai2d',
            f'./{model_slug}/chartqa',
            f'./{model_slug}/textvqa',
            f'./{model_slug}/ocrbench',
        ],
        'human_scores': {
            'scienceqa': 88.4, 'mmmu': 88.6, 'mmmupro': 85.6, 'gqa': 89.3,
            'vizwiz': 75.0, 'mmvp': 95.7, 'qbench': 81.74, 'blink': 95.7, 'vstar': 98.95,
        },
        'group1': {'coco', 'realworldqa', 'pope', 'ade', 'mmvp', 'qbench', 'vstar', 'blink', 'omni', 'mmstar'},
        'group2': {'ai2d', 'textvqa', 'chartqa', 'ocrbench'},
        'output_suffix': '_g2',
    }

    for label, cfg in [('g1', config_g1), ('g2', config_g2)]:
        print(f"\n=== Plot {label} ===")
        accuracy_data = load_accuracy_data(cfg['dataset_dirs'])
        plot_radar_chart(
            accuracy_data,
            model_slug,
            cfg['human_scores'],
            cfg['group1'],
            cfg['group2'],
            cfg['output_suffix'],
        )


if __name__ == '__main__':
    main()
