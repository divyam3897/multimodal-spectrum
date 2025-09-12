#!/usr/bin/env python3
"""
Accuracy Comparison Script for Vision-Language Model Evaluation
Parses all summary JSON files and creates comprehensive comparisons across shuffle conditions.
Works with any dataset (MME, VQA, etc.).
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from collections import defaultdict
import numpy as np

def load_summary_files(answers_dir, dataset_name=None):
    """Load all summary JSON files from the answers directory."""
    # Look for summary files with the new naming pattern
    if dataset_name:
        # Look for dataset-specific summary files
        summary_pattern = os.path.join(answers_dir, f"summary_*_{dataset_name}_*.json")
        summary_files = glob.glob(summary_pattern)
        if not summary_files:
            # Fallback to general pattern
            summary_pattern = os.path.join(answers_dir, "summary_*.json")
            summary_files = glob.glob(summary_pattern)
    else:
        # General pattern
        summary_pattern = os.path.join(answers_dir, "summary_*.json")
        summary_files = glob.glob(summary_pattern)
    
    summaries = []
    
    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['file_path'] = file_path
                data['filename'] = os.path.basename(file_path)
                summaries.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return summaries

def create_comparison_plots(summaries, output_dir, dataset_name="Dataset"):
    """Create comprehensive comparison plots."""
    
    # Organize data by shuffle conditions using answers_file information
    shuffle_conditions = defaultdict(list)
    for summary in summaries:
        # Extract condition from answers_file path or filename
        answers_file = summary.get('answers_file', '')
        filename = summary.get('filename', '')
        
        # Check both answers_file and summary filename for condition indicators
        condition_text = (answers_file + ' ' + filename).lower()
        
        if 'nrm' in condition_text or 'normal' in condition_text:
            condition = "Normal"
        elif 'txt' in condition_text and 'img' not in condition_text:
            condition = "Text Shuffle"
        elif 'img' in condition_text and 'txt' not in condition_text:
            condition = "Image Shuffle"
        elif ('txt' in condition_text and 'img' in condition_text) or 'rdm' in condition_text or 'random' in condition_text:
            condition = "Both Shuffles"
        else:
            # Fallback to shuffle_settings if filename doesn't match pattern
            text_shuffle = summary.get('shuffle_settings', {}).get('text_shuffle', False)
            image_shuffle = summary.get('shuffle_settings', {}).get('image_shuffle', False)
            
            if not text_shuffle and not image_shuffle:
                condition = "Normal"
            elif text_shuffle and not image_shuffle:
                condition = "Text Shuffle"
            elif not text_shuffle and image_shuffle:
                condition = "Image Shuffle"
            else:
                condition = "Both Shuffles"
        
        shuffle_conditions[condition].append(summary)
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'{dataset_name} Evaluation: Accuracy Comparison Across Shuffle Conditions', fontsize=20, fontweight='bold')
    
    # Plot 1: Overall Accuracy Comparison
    conditions = list(shuffle_conditions.keys())
    overall_accuracies = []
    
    for condition in conditions:
        if shuffle_conditions[condition]:
            # Average if multiple chunks
            avg_acc = np.mean([s['overall_accuracy'] for s in shuffle_conditions[condition]])
            overall_accuracies.append(avg_acc)
        else:
            overall_accuracies.append(0)
    
    bars1 = axes[0, 0].bar(conditions, overall_accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'], alpha=0.8)
    axes[0, 0].set_title('Overall Accuracy by Shuffle Condition', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars1, overall_accuracies):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Per-Category Accuracy Heatmap
    # Get all categories
    all_categories = set()
    for summary in summaries:
        all_categories.update(summary['category_accuracies'].keys())
    all_categories = sorted(list(all_categories))
    
    # Create accuracy matrix
    accuracy_matrix = []
    condition_labels = []
    
    for condition in conditions:
        if shuffle_conditions[condition]:
            condition_labels.append(condition)
            # Average across chunks if multiple
            category_accs = defaultdict(list)
            for summary in shuffle_conditions[condition]:
                for cat, acc_data in summary['category_accuracies'].items():
                    category_accs[cat].append(acc_data['accuracy'])
            
            # Get average accuracy per category
            row = []
            for cat in all_categories:
                if cat in category_accs:
                    row.append(np.mean(category_accs[cat]))
                else:
                    row.append(0)
            accuracy_matrix.append(row)
    
    if accuracy_matrix:
        im = axes[0, 1].imshow(accuracy_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, 1].set_title('Per-Category Accuracy Heatmap', fontweight='bold', fontsize=14)
        axes[0, 1].set_xticks(range(len(all_categories)))
        axes[0, 1].set_xticklabels(all_categories, rotation=45, ha='right')
        axes[0, 1].set_yticks(range(len(condition_labels)))
        axes[0, 1].set_yticklabels(condition_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0, 1])
        cbar.set_label('Accuracy', fontweight='bold')
        
        # Add text annotations
        for i in range(len(condition_labels)):
            for j in range(len(all_categories)):
                text = axes[0, 1].text(j, i, f'{accuracy_matrix[i][j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    # Plot 3: Sample Distribution
    if shuffle_conditions['Normal']:  # Use normal condition for sample distribution
        normal_summary = shuffle_conditions['Normal'][0]
        categories = list(normal_summary['category_accuracies'].keys())
        sample_counts = [normal_summary['category_accuracies'][cat]['total'] for cat in categories]
        
        bars3 = axes[1, 0].bar(range(len(categories)), sample_counts, color='lightsteelblue', alpha=0.8)
        axes[1, 0].set_title('Sample Distribution per Category', fontweight='bold', fontsize=14)
        axes[1, 0].set_xlabel('Categories', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Samples', fontweight='bold')
        axes[1, 0].set_xticks(range(len(categories)))
        axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars3, sample_counts):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Accuracy Drop Analysis
    if 'Normal' in shuffle_conditions and len(shuffle_conditions) > 1:
        normal_acc = shuffle_conditions['Normal'][0]['category_accuracies']
        
        accuracy_drops = {}
        for condition in conditions:
            if condition != 'Normal' and shuffle_conditions[condition]:
                shuffle_acc = shuffle_conditions[condition][0]['category_accuracies']
                drops = []
                for cat in normal_acc.keys():
                    if cat in shuffle_acc:
                        drop = normal_acc[cat]['accuracy'] - shuffle_acc[cat]['accuracy']
                        drops.append(drop)
                accuracy_drops[condition] = np.mean(drops) if drops else 0
        
        if accuracy_drops:
            conditions_drop = list(accuracy_drops.keys())
            drops = list(accuracy_drops.values())
            colors = ['lightcoral' if d > 0 else 'lightgreen' for d in drops]
            
            bars4 = axes[1, 1].bar(conditions_drop, drops, color=colors, alpha=0.8)
            axes[1, 1].set_title('Average Accuracy Drop from Normal', fontweight='bold', fontsize=14)
            axes[1, 1].set_ylabel('Accuracy Drop', fontweight='bold')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, drop in zip(bars4, drops):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                               f'{drop:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Minimize white space
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.92, wspace=0.25, hspace=0.3)
    
    # Save only as PDF with dataset-specific filename
    dataset_suffix = f"_{dataset_name.lower()}" if dataset_name else ""
    comparison_file = os.path.join(output_dir, f"accuracy_comparison_all_shuffles{dataset_suffix}")
    plt.savefig(f"{comparison_file}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return comparison_file

def create_summary_table(summaries, output_dir, dataset_name="dataset"):
    """Create a summary table in CSV and JSON format."""
    
    table_data = []
    for summary in summaries:
        text_shuffle = "ON" if summary.get('shuffle_settings', {}).get('text_shuffle', False) else "OFF"
        image_shuffle = "ON" if summary.get('shuffle_settings', {}).get('image_shuffle', False) else "OFF"
        
        row = {
            'Model': summary['model_name'],
            'Answers_File': summary.get('answers_file', 'Unknown'),
            'Text_Shuffle': text_shuffle,
            'Image_Shuffle': image_shuffle,
            'Overall_Accuracy': summary['overall_accuracy'],
            'Total_Questions': summary['total_questions'],
            'Total_Correct': summary['total_correct'],
            'Chunk_Idx': summary.get('chunk_info', {}).get('chunk_idx', 0)
        }
        
        # Add per-category accuracies
        for cat, acc_data in summary['category_accuracies'].items():
            row[f'{cat}_Accuracy'] = acc_data['accuracy']
            row[f'{cat}_Total'] = acc_data['total']
        
        table_data.append(row)
    
    # Save as CSV with dataset-specific filename
    dataset_suffix = f"_{dataset_name.lower()}" if dataset_name else ""
    df = pd.DataFrame(table_data)
    csv_file = os.path.join(output_dir, f"accuracy_summary_table{dataset_suffix}.csv")
    df.to_csv(csv_file, index=False)
    
    # Save as JSON
    json_file = os.path.join(output_dir, f"accuracy_summary_table{dataset_suffix}.json")
    with open(json_file, 'w') as f:
        json.dump(table_data, f, indent=2)
    
    return csv_file, json_file

def main():
    parser = argparse.ArgumentParser(description='Compare accuracy across all shuffle conditions')
    parser.add_argument('--answers_dir', type=str, required=True, help='Directory containing summary JSON files')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (defaults to answers_dir)')
    parser.add_argument('--dataset_name', type=str, help='Dataset name (optional)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.answers_dir
    
    # Load all summary files
    summaries = load_summary_files(args.answers_dir, args.dataset_name)
    
    if not summaries:
        print(f"No summary files found in {args.answers_dir}")
        return
    
    print(f"Found {len(summaries)} summary files")
    
    # Create comparison plots
    comparison_file = create_comparison_plots(summaries, args.output_dir, args.dataset_name)
    print(f"Created comparison plots: {comparison_file}.pdf")
    
    # Create summary table
    csv_file, json_file = create_summary_table(summaries, args.output_dir, args.dataset_name)
    print(f"Created summary table: {csv_file}")
    print(f"Created summary JSON: {json_file}")
    
    print("\n=== COMPARISON COMPLETE ===")
    print(f"All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 