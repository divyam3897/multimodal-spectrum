import json
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

EVAL_TYPE_MAPPING = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}

REVERSE_EVAL_MAPPING = {item: key for key, values in EVAL_TYPE_MAPPING.items() for item in values}

def detect_condition_from_filename(filename: str) -> str:
    filename_lower = os.path.basename(filename).lower()
    if 'nrm' in filename_lower or 'normal' in filename_lower or ('txtoff' in filename_lower and 'imgoff' in filename_lower):
        return "Normal"
    if 'txt' in filename_lower and 'img' not in filename_lower or ('txton' in filename_lower and 'imgoff' in filename_lower):
        return "Text Shuffle"
    if 'img' in filename_lower and 'txt' not in filename_lower or ('imgaon' in filename_lower and 'txtoff' in filename_lower):
        return "Image Shuffle"
    if 'rdm' in filename_lower or 'random' in filename_lower or ('txton' in filename_lower and 'imgon' in filename_lower):
        return "Random"
    print(f"Warning: Could not detect condition from filename '{filename}'. Defaulting to 'Unknown'.")
    return "Unknown"

def calculate_metrics_from_file(jsonl_file: str) -> dict:
    model_id = ""
    category_stats = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line {i+1} in {jsonl_file}")
                continue
            model_id = data.get('model_id', model_id)
            category = data.get('category')
            if not category: continue
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'acc_plus_running_total': 0, 'total': 0}
            gt_answer = str(data.get('gt_answer', '')).lower().strip().rstrip('.')
            answer = str(data.get('answer', '')).lower().strip().rstrip('.')
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
            'accuracy': accuracy * 100,
            'accuracy_plus': acc_plus * 100,
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
        'model_name': model_id,
        'overall_metrics': {
            'total_score': perception_score + cognition_score,
            'perception_score': perception_score, 'cognition_score': cognition_score,
            'overall_accuracy': overall_accuracy * 100, 
            'overall_accuracy_plus': overall_acc_plus * 100, 
            'total_questions': total_count, 'total_correct': total_correct,
        },
        'category_scores': category_scores
    }


def save_comparison_results(all_results: dict, output_dir: str):
    if not all_results:
        print("No results to save.")
        return
    model_name = next(iter(all_results.values()))['model_name']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"mme_comparison_{model_slug}.json")
    
    sorted_conditions = {}
    for cond, res in all_results.items():
        sorted_category_scores = dict(sorted(res['category_scores'].items()))
        sorted_conditions[cond] = {
            'overall_metrics': res['overall_metrics'], 
            'category_scores': sorted_category_scores
        }
    
    output_data = {
        'model_name': model_name,
        'generated_at': datetime.now().isoformat(),
        'conditions': sorted_conditions,
    }
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n Saved comprehensive JSON results to: {json_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--compare_dir", type=str)
    args = parser.parse_args()
    
    output_dir = args.compare_dir
    os.makedirs(output_dir, exist_ok=True)

    jsonl_pattern = os.path.join(args.compare_dir, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    if not jsonl_files:
        print(f"Error: No files found matching '{jsonl_pattern}'.")
    else:
        print(f"Found {len(jsonl_files)} files for comparison in '{args.compare_dir}'.")
        all_results = {}
        for f in jsonl_files:
            condition = detect_condition_from_filename(f)
            print(f"  - Processing '{os.path.basename(f)}' (Condition: {condition})")
            all_results[condition] = calculate_metrics_from_file(f)
        
        save_comparison_results(all_results, output_dir)