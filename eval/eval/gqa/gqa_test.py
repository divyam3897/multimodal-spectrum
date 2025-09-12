import os
import json
import nltk
import csv
import argparse
import glob
from datetime import datetime


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


def is_inflection(word1, word2):
    """
    Checks if two words are likely inflections of the same word using stemming and lemmatization.

    Args:
        word1: The first word.
        word2: The second word.

    Returns:
        True if the words are likely inflections, False otherwise.
    """
    # Lowercase both words for case-insensitive comparison
    word1 = word1.lower()
    word2 = word2.lower()

    # Use Porter stemmer for a more aggressive reduction to the base form
    stemmer = nltk.PorterStemmer()
    stem1 = stemmer.stem(word1)
    stem2 = stemmer.stem(word2)

    # Use WordNet lemmatizer for a more accurate base form considering context
    lemmatizer = nltk.WordNetLemmatizer()
    lemma1 = lemmatizer.lemmatize(word1)
    lemma2 = lemmatizer.lemmatize(word2)

    # Check if stemmed or lemmatized forms are equal
    return (stem1 == stem2) or (lemma1 == lemma2)


def calculate_metrics_from_file(jsonl_file: str) -> dict:
    correct, total = 0, 0
    model = ""
    with open(jsonl_file, 'r') as file:
        for line in file:
            total += 1.0
            data = json.loads(line)
            model = data.get('model_id', '')
            answer = data.get('answer', '').lower().strip().rstrip('.')
            gt_answer = data.get('gt_answer', '').lower().strip().rstrip('.')
            if (answer == gt_answer):
                correct += 1.0
           

    return {
        "model": model,
        "time": time_string,
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct/total,
    }

def save_comparison_results(all_results: dict, output_dir: str):
    if not all_results:
        print("No results to save.")
        return
    print(all_results.values())
    model_name = next(iter(all_results.values()))['model']
    model_slug = model_name.replace('/', '_').replace('-', '_')
    json_output_path = os.path.join(output_dir, f"gqa_comparison_{model_slug}.json")
    
    # Sort category_scores alphabetically by category name for each condition
    sorted_conditions = {}
    for cond, res in all_results.items():
        # sorted_category_scores = dict(sorted(res['category_scores'].items()))
        print(res)
        sorted_conditions[cond] = {
            'overall_metrics': res['accuracy'], 
            # 'category_scores': sorted_category_scores
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