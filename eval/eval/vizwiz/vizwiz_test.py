import os
import json
import csv
from datetime import datetime
import argparse
import glob

from m4c_evaluator import TextVQAAccuracyEvaluator

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

def calculate_metrics_from_file(jsonl_file: str) -> dict:
    pred_list = []
    test_list = []

    model = ""
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            answer = data.get('answer', '').lower().strip().rstrip('.')
            image_id = data.get('image_id', '')
            model = data.get("model_id", '')
            gt_answer = [x.lower().strip().rstrip('.') for x in data.get('gt_answer', [''])]
            category = data.get('category', '')
            if category == "val":
                pred_list.append({
                    "pred_answer": answer,
                    "gt_answers": gt_answer,
                })
            else:
                test_list.append({
                    "image": image_id,
                    "answer": answer
                })
    
    evaluator = TextVQAAccuracyEvaluator()
    return {
        "model": model,
        "time": time_string,
        "accuracy": 100. * evaluator.eval_pred_list(pred_list),
    }
    

def save_comparison_results(all_results: dict, output_dir: str):
    if not all_results:
        print("No results to save.")
        return
    print(all_results.values())
    model_name = next(iter(all_results.values()))['model']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"vizwiz_comparison_{model_slug}.json")
    
    sorted_conditions = {}
    for cond, res in all_results.items():
        print(res)
        sorted_conditions[cond] = {
            'overall_metrics': res['accuracy'], 
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