import os
import json
import csv
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

def edit_distance(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    for i in range(len(str1) + 1):
        dp[i][0] = i
    for j in range(len(str2) + 1):
        dp[0][j] = j

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

    return dp[len(str1)][len(str2)]


def get_accuracy(pred_list, extra_penalization):
    correct = 0.0
    for items in pred_list:
        pred_answer = items["pred_answer"]
        gt_answers = [x for x in items["gt_answers"]]
        found = False
        if extra_penalization:
            for x in gt_answers:
                if edit_distance(pred_answer, x) == 0:
                    found = True
                    break
        else:
            for x in gt_answers:
                if x in pred_answer:
                    found = True
                    break
        if found:
            correct += 1.0
    return correct/len(pred_list)


def calculate_metrics_from_file(jsonl_file: str, extra_penalization: bool = True) -> dict:
    pred_list = {}
    full_list = []
    model = ""
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            model = data.get('model_id', '')
            category = data.get('category', '')

            if category == "Handwritten Mathematical Expression Recognition":
                answer = data.get('answer', '').strip().replace('\n', ' ').replace(' ', '')
                gt_answer = [x.strip().replace('\n', ' ').replace(' ', '') for x in data.get('gt_answer', [''])]
            else:
                answer = data.get('answer', '').lower().strip().replace('\n', ' ')
                gt_answer = [x.lower().strip().replace('\n', ' ') for x in data.get('gt_answer', [''])]

            full_list.append({
                    "pred_answer": answer,
                    "gt_answers": gt_answer,
                })
            if category in pred_list:
                pred_list[category].append({
                    "pred_answer": answer,
                    "gt_answers": gt_answer,
                })
            else:
                pred_list[category] = [{
                    "pred_answer": answer,
                    "gt_answers": gt_answer,
                }]

    combined_data = {
        "model": model,
        "time": time_string,
    }
    for category, preds in pred_list.items():
        combined_data[category] = 100.0 * get_accuracy(preds, extra_penalization)

    combined_data['accuracy'] = 100.0 * get_accuracy(full_list, extra_penalization)
    return combined_data

def save_comparison_results(all_results: dict, output_dir: str):
    if not all_results:
        print("No results to save.")
        return
    print(all_results.values())
    model_name = next(iter(all_results.values()))['model']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"ocrbench_comparison_{model_slug}.json")
    
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