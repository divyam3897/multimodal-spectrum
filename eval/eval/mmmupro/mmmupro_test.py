import os
import json
import csv
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

def extract_mcq_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    
    text = text.strip().rstrip('.:,').lstrip('(').rstrip(')')
    if len(text) > 1:
        text = text[0]
    return text

def extract_single_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    text = text.strip().rstrip('.')
    return text

def relaxed_accuracy(pred, gt):
    return 1 if abs(pred-gt) <= abs(gt)*0.05 else 0


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def detect_condition_from_filename(filename: str) -> str:
    """Detect experimental condition from filename"""
    filename_lower = os.path.basename(filename).lower()
    if 'nrm' in filename_lower or 'normal' in filename_lower or ('txtoff' in filename_lower and 'imgoff' in filename_lower):
        return "Normal"
    if 'txt' in filename_lower and 'img' not in filename_lower or ('txton' in filename_lower and 'imgoff' in filename_lower):
        return "Text Shuffle"
    if 'img' in filename_lower and 'txt' not in filename_lower or ('imgon' in filename_lower and 'txtoff' in filename_lower):
        return "Image Shuffle"
    if 'rdm' in filename_lower or 'random' in filename_lower or ('txton' in filename_lower and 'imgon' in filename_lower):
        return "Random"
    print(f"Warning: Could not detect condition from filename '{filename}'. Defaulting to 'Unknown'.")
    return "Unknown"


def calculate_metrics_from_file(jsonl_file: str) -> dict:
    """Calculate metrics from a single JSONL file"""
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
            if not category: 
                continue
                
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0}
            
            # MMMU-Pro is all multiple choice with 10 options
            answer = extract_mcq_answer(data.get('answer', ''))
            gt_answer = data.get('gt_answer', '').lower()
            
            stats = category_stats[category]
            stats['total'] += 1
            
            match = False
            if answer == gt_answer:
                stats['correct'] += 1
                match = True
            elif is_number(gt_answer) and is_number(answer) and relaxed_accuracy(float(gt_answer), float(answer)):
                stats['correct'] += 1
                match = True

    # Calculate category scores
    category_scores = {}
    total_correct, total_count = 0, 0
    
    for category, stats in category_stats.items():
        total = stats['total']
        if total == 0: 
            continue
            
        accuracy = stats['correct'] / total
        category_scores[category] = {
            'accuracy': accuracy,
            'score': accuracy * 100,  # Simple score for MMMU-Pro
            'correct': stats['correct'],
            'total': total,
        }
        total_correct += stats['correct']
        total_count += total

    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0
    
    return {
        'model_name': model_id,
        'overall_metrics': {
            'total_score': overall_accuracy * 100,
            'overall_accuracy': overall_accuracy * 100,
            'total_questions': total_count,
            'total_correct': total_correct,
        },
        'category_scores': category_scores
    }


def save_comparison_results(all_results: dict, output_dir: str):
    """Save comprehensive comparison results to JSON"""
    if not all_results:
        print("No results to save.")
        return
        
    model_name = next(iter(all_results.values()))['model_name']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"mmmupro_comparison_{model_slug}.json")
    
    # Sort category_scores alphabetically by category name for each condition
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
    print(f"\n✅ Saved comprehensive JSON results to: {json_output_path}")


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    """Original single-file metrics computation (backward compatibility)"""
    model = ""
    categories = set()  # To store unique categories
    category_metrics = {}  # To store metrics for each category

    with open(jsonl_file, 'r') as file:
        output_file = os.path.expanduser(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for line in file:
                data = json.loads(line)
                model = data.get('model_id', '')
                category = data.get('category', '')
                categories.add(category) 

                if category not in category_metrics:
                    category_metrics[category] = {'matches': 0, 'total': 0}

                # MMMU-Pro is all multiple choice (no need to check type)
                answer = extract_mcq_answer(data.get('answer', ''))
                gt_answer = data.get('gt_answer', '').lower()
                category_metrics[category]['total'] += 1

                match = False
                if answer == gt_answer:
                    category_metrics[category]['matches'] += 1
                    match = True
                elif is_number(gt_answer) and is_number(answer) and relaxed_accuracy(float(gt_answer), float(answer)):
                    category_metrics[category]['matches'] += 1
                    match = True
                else:
                    pass

                if not match:
                    out_file.write(line)

    category_scores = {}
    total_matches = 0
    total_count = 0

    for category, metrics in category_metrics.items():
        matches = metrics['matches']
        total = metrics['total']

        total_matches += matches
        total_count += total

        accuracy = (matches * 1.0 / total)

        category_scores[category] = {'accuracy': accuracy, 'total': total}

    overall_accuracy = (total_matches * 1.0 / total_count)

    overall_metrics = {
        'accuracy': overall_accuracy,
        'total_count': total_count
    }

    combined_data = {
        "model": model,
        "time": time_string,
        **overall_metrics,
        **category_scores
    }

    add_data_to_csv(csv_file, combined_data)
    print(f"Saved {model} metrics to {csv_file}")

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"mmmupro_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Added a copy of the csv file to {extra_csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--answers_file", type=str, help="Path to single jsonl file (backward compatibility)")
    mode_group.add_argument("--compare_dir", type=str, help="Path to directory containing multiple jsonl files for S-metrics comparison")
    
    # Original arguments (for backward compatibility)
    parser.add_argument("--output_file", type=str, default="./incorrect/incorrect.jsonl", help="Path to output file for incorrect predictions")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to output csv file")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Extra output directory")

    args = parser.parse_args()
    
    if args.answers_file:
        # Original single-file mode
        compute_metrics(args.answers_file, args.output_file, args.csv_file, args.extra_outdir)
    
    elif args.compare_dir:
        # New S-metrics comparison mode
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
