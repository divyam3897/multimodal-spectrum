import os
import json
import csv
import glob
from datetime import datetime

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)
    print(f"    Writing CSV to: {file_path}")
    print(f"    File exists: {file_exists}")
    print(f"    Data keys: {list(data.keys())}")

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()
            print(f"    Wrote header")

        writer.writerow(data)
        print(f"    Wrote data row")


def extract_model_slug_from_path(directory_path):
    path_parts = directory_path.split('/')
    for part in path_parts:
        if part.startswith('answers_'):
            return part.replace('answers_', '')
    return os.path.basename(directory_path.rstrip('/'))


def validate_docvqa_submission(test_list):
    """Validate that the submission format is correct for DocVQA Task 2"""
    if not test_list:
        print("    Warning: No test data found!")
        return False
    
    sample_entry = test_list[0]
    required_fields = ['question_id', 'evidence', 'answer']
    
    for field in required_fields:
        if field not in sample_entry:
            print(f"    Error: Missing required field '{field}' in submission data")
            return False
    
    for i, entry in enumerate(test_list[:5]):  
        if not isinstance(entry['question_id'], int):
            print(f"    Error: question_id must be an integer, got {type(entry['question_id'])}")
            return False
        if not isinstance(entry['evidence'], list):
            print(f"    Error: evidence must be a list, got {type(entry['evidence'])}")
            return False
        if not isinstance(entry['answer'], list):
            print(f"    Error: answer must be a list, got {type(entry['answer'])}")
            return False
    
    print(f"    ✓ Submission format validated: {len(test_list)} questions")
    return True


def compute_metrics(output_dir):
    model_slug = extract_model_slug_from_path(output_dir)
    print(f"Extracted model slug: {model_slug}")
    
    jsonl_pattern = os.path.join(output_dir, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    if not jsonl_files:
        print(f"Error: No .jsonl files found in '{output_dir}'")
        return
    
    print(f"Found {len(jsonl_files)} .jsonl files to process:")
    for jsonl_file in jsonl_files:
        print(f"  - {os.path.basename(jsonl_file)}")
    
    for answers_file in jsonl_files:
        print(f"\nProcessing: {os.path.basename(answers_file)}")
        
        pred_list = []
        test_list = []
        model = ""
        
        with open(answers_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                question_id = data.get('question_id', data.get('questionId', ''))
                answer = data.get('answer', '')
                evidence = data.get('evidence', [])
                model = data.get("model_id", '')
                
                if not isinstance(answer, list):
                    answer = [answer] if answer else []
                
                if not isinstance(evidence, list):
                    evidence = [evidence] if evidence else []
                
                test_list.append({
                    "question_id": int(question_id),
                    "evidence": evidence,
                    "answer": answer
                })
        
        validate_docvqa_submission(test_list)
        
        base_name = os.path.splitext(os.path.basename(answers_file))[0]
        file_path = os.path.join(output_dir, f"result_task2_{base_name}_{model_slug}.json")
        
        with open(file_path, "w") as json_file:
            json.dump(test_list, json_file, indent=2)
        
        combined_data = {
            "model_slug": model_slug,
            "model_id": model,
            "file_name": base_name,
            "submission_file": f"result_task2_{base_name}_{model_slug}.json",
            "total_questions": len(test_list),
            "time": time_string,
            "accuracy": "add here after submission",
            "submission_url": "https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=2",
        }
        
        print(f"  CSV data: {combined_data}")
        
        csv_file = os.path.join(output_dir, f"docvqa_submission_{base_name}_{model_slug}.csv")
        add_data_to_csv(csv_file, combined_data)
        
        print(f"  Created submission file: {file_path}")
        print(f"  Created CSV file: {csv_file}")
    
    with open("./docvqa_submission_url.txt", "w") as f:
        f.write("https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=2")
    
    print(f"\nAll files processed!")
    print(f"Submission files created in: {output_dir}")
    print(f"To submit to DocVQA portal:")
    print(f"1. Go to: https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=2")
    print(f"2. Upload the JSON files: result_task2_*_{model_slug}.json")
    print(f"3. For more info, visit: https://www.docvqa.org/")
    print(f"4. Contact: docvqa@cvc.uab.es for questions")
    print(f"\nSubmission format:")
    print(f"- Each entry should have: question_id (int), evidence (list of relevance scores), answer (list of values)")
    print(f"- Evidence scores should represent confidence that document contains the answer")
    print(f"- Answer should contain values extracted from positive evidence documents")

    # add a duplicate copy of the info to the extra_outdir
    # if extra_outdir is not None:
    #     os.makedirs(extra_outdir, exist_ok=True)
    #     extra_submission_file = os.path.join(extra_outdir, f"docvqa_submission_{model}.json")
    #     with open(extra_submission_file, "w") as json_file:
    #         json.dump(test_list, json_file)
    #     with open(os.path.join(extra_outdir, "docvqa_submission_url.txt"), "w") as f:
    #         f.write("https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=2")

    #     print(f"Added a copy of the submission file to {extra_submission_file}")

    #     extra_csv_file = os.path.join(extra_outdir, f"docvqa_{model}.csv")
    #     add_data_to_csv(extra_csv_file, combined_data)
    #     print(f"Added a copy of the submission info to {extra_csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_dir", type=str, required=True, help="Path to an extra output directory in which to store a copy of the information")
   
    args = parser.parse_args()

    compute_metrics(args.compare_dir)
