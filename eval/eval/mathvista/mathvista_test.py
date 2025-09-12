import json
import csv
import os
from datetime import datetime
import argparse
import glob

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


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

def relaxed_accuracy(pred, gt):
    return 1 if abs(pred-gt) <= abs(gt)*0.05 else 0


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def calculate_metrics_from_file(jsonl_file):
    model = ""
    categories = set()  # To store unique categories
    task_metrics = {}  # To store metrics for each task

    with open(jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                model = data.get('model_id', '')
                task = data.get('task', '')
                categories.add(task) 

                if task not in task_metrics:
                    task_metrics[task] = {'matches': 0, 'total': 0}

                answer = data.get('answer', '').lower().strip()
                if data.get('type', '') == "multiple-choice":
                    answer = answer.split('.')[0]
                gt_answer = data.get('gt_answer', '').lower()
                task_metrics[task]['total'] += 1

                if answer == gt_answer:
                    task_metrics[task]['matches'] += 1
                elif is_number(gt_answer) and is_number(answer) and relaxed_accuracy(float(gt_answer), float(answer)):
                    task_metrics[task]['matches'] += 1

    task_scores = {}
    total_matches = 0
    total_count = 0

    for task, metrics in task_metrics.items():
        matches = metrics['matches']
        total = metrics['total']

        total_matches += matches
        total_count += total

        accuracy = (matches * 1.0 / total)

        task_scores[task] = {'accurcay': accuracy * 100, 'total': total}

    overall_accuracy = (total_matches * 1.0 / total_count)

    overall_metrics = {
        'accuracy': overall_accuracy * 100,
        'total_count': total_count
    }

    combined_data = {
        "model": model,
        "time": time_string,
    }
    combined_data.update(overall_metrics)
    combined_data.update(task_scores)
    
    return combined_data

def save_comparison_results(all_results: dict, output_dir: str):
    if not all_results:
        print("No results to save.")
        return
    print(all_results.values())
    model_name = next(iter(all_results.values()))['model']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"mathvista_comparison_{model_slug}.json")
    
    # Sort category_scores alphabetically by category name for each condition
    sorted_conditions = {}
    for cond, res in all_results.items():
        # Extract category scores from the result (they are direct keys, not nested)
        category_scores = {}
        for key, value in res.items():
            if key not in ['model', 'time', 'accuracy', 'total_count']:
                category_scores[key] = value
        
        sorted_category_scores = dict(sorted(category_scores.items()))
        print(res)
        sorted_conditions[cond] = {
            'overall_metrics': res['accuracy'], 
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