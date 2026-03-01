import json
import csv
import os
import string
from datetime import datetime
import argparse
import glob

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_sentence(sentence):
    translator = str.maketrans('', '', string.punctuation)
    normalized_sentence = sentence.translate(translator).lower().strip()
    return normalized_sentence

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


def compare_sentences(sentence1, sentence2):
    return normalize_sentence(sentence1) == normalize_sentence(sentence2)


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def is_correct(answer, gt_answer, text_answer):
    option = answer.split('.')[0]
    if compare_sentences(answer, text_answer):
        return True
    elif option == gt_answer:
        return True
    else:
        return False

def extract_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    text = text.strip().rstrip('.')
    return text


def calculate_metrics_from_file(jsonl_file: str) -> dict:
    model = ""
    categories = set()  # To store unique categories
    category_metrics = {}  # To store metrics for each category
    category_metrics["is_multimodal"] = {'matches': 0, 'total': 0}

    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            model = data.get('model_id', '')
            category = data.get('category', '')
            # Support multiple schemas: legacy uses 'type', new evals may not provide it
            qn_type = data.get('type', False)
            categories.add(category)

            if category not in category_metrics:
                category_metrics[category] = {'matches': 0, 'total': 0}

            # Support both legacy and new keys
            answer_field = data.get('answer', data.get('model_output', ''))
            answer = extract_answer(answer_field)
            gt_answer = str(data.get('gt_answer', data.get('ground_truth_answer', ''))).lower().strip()
            text_answer = data.get('text_answer', data.get('ground_truth_text', '')).lower().strip()
            category_metrics[category]['total'] += 1
            if qn_type:
                category_metrics['is_multimodal']['total'] += 1

            if is_correct(answer, gt_answer, text_answer):
                category_metrics[category]['matches'] += 1
            if qn_type:
                category_metrics['is_multimodal']['matches'] += 1

    category_total_scores = {}
    total_matches = 0
    total_count = 0

    for category, metrics in category_metrics.items():
        matches = metrics['matches']
        total = metrics['total']

        total_matches += matches
        total_count += total

        # Avoid division by zero
        accuracy = (matches * 1.0 / total) * 100 if total > 0 else 0.0

        category_total_scores[category] = {'accuracy': accuracy, 'total': total}

    multimodal_accuracy = category_total_scores.get("is_multimodal", {"accuracy": 0.0})["accuracy"]

    overall_accuracy = (total_matches * 1.0 / total_count) if total_count > 0 else 0.0

    overall_metrics = {
        'accuracy': overall_accuracy * 100,
        'total_count': total_count
    }

    combined_data = {
        "model": model,
        "time": time_string,
        "multimodal_acc": multimodal_accuracy * 100,
    }
    combined_data.update(overall_metrics)
    combined_data.update(category_total_scores)
    return combined_data

def save_comparison_results(all_results: dict, output_dir: str):
    if not all_results:
        print("No results to save.")
        return
    print(all_results.values())
    model_name = next(iter(all_results.values()))['model']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"scienceqa_comparison_{model_slug}.json")
    
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