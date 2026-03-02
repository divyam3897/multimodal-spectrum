import os
import json
import csv
from datetime import datetime
import argparse
import glob
import re

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


def extract_mcq_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text.split(answer_keyword)[-1]
    
    text = text.strip().rstrip('.').lstrip('(').rstrip(')').strip()
    
    letter_match = re.search(r'^([a-d])', text)
    if letter_match:
        return letter_match.group(1)
   
    if len(text) > 1:
        text = text[0]
    
    return text

def calculate_metrics_from_file(jsonl_file: str) -> dict:
    model = ""
    categories = set()  
    category_metrics = {} 

    with open(jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                model = data.get('model_id', '')
                category = data.get('category', '')
                categories.add(category) 

                if category not in category_metrics:
                    category_metrics[category] = {'matches': 0, 'total': 0}

                answer = extract_mcq_answer(data.get('answer', ''))
                gt_answer = extract_mcq_answer(data.get('gt_answer', ''))  

                category_metrics[category]['total'] += 1

                if answer == gt_answer:
                    category_metrics[category]['matches'] += 1

    category_scores = {}
    total_matches = 0
    total_count = 0

    for category, metrics in category_metrics.items():
        matches = metrics['matches']
        total = metrics['total']

        total_matches += matches
        total_count += total

        accuracy = (matches * 1.0 / total)

        category_scores[category] = {'accuracy': accuracy * 100, 'total': total}

    overall_accuracy = (total_matches * 1.0 / total_count)

    overall_metrics = {
        'accuracy': overall_accuracy * 100,
        'total_count': total_count
    }

    return {
        "model": model,
        "time": time_string,
        **overall_metrics,
        **category_scores
    }




def save_comparison_results(all_results: dict, output_dir: str):
    if not all_results:
        print("No results to save.")
        return
    print(all_results.values())
    model_name = next(iter(all_results.values()))['model']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"blink_comparison_{model_slug}.json")
    
    sorted_conditions = {}
    for cond, res in all_results.items():
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