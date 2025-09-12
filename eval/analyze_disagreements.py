import argparse
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import textwrap
import random

def load_jsonl(file_path):
    """Loads a .jsonl file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def wrap_text(text, width=60):
    """Wraps text to a specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))

def save_analysis_image(case_info, output_dir, dataset_name, case_type):
    """
    Saves a comprehensive visualization for a single analysis case.
    """
    # Create the detailed output directory
    category_dir = os.path.join(output_dir, dataset_name, case_type, case_info['category'])
    os.makedirs(category_dir, exist_ok=True)
    
    qid = case_info['qid']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Analysis Case (QID: {qid}) | Category: {case_info['category']}\n{case_type.replace('_', ' ').title()} vs. Normal", fontsize=18, fontweight='bold')

    # --- Panel 1: Original Image and Question ---
    ax = axes[0, 0]
    if case_info["image"] is not None:
        ax.imshow(case_info["image"])
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
    ax.set_title('1. Original Input', fontweight='bold', fontsize=12)
    ax.axis('off')
    orig_text = f"Q: {wrap_text(case_info['question'])}\n\nGround Truth: {case_info['gt_answer']}"
    ax.text(0, -0.1, orig_text, transform=ax.transAxes, fontsize=10, va='top')

    # --- Panel 2: Model Answers ---
    ax = axes[0, 1]
    ax.set_title('2. Model Answers', fontweight='bold', fontsize=12)

    nrm_result = 'CORRECT' if case_info['nrm_correct'] else 'INCORRECT'
    alt_result = 'CORRECT' if case_info['alt_correct'] else 'INCORRECT'
    
    alt_model_name = "Text Shuffle" if "text" in case_type else "Image Shuffle"

    answers_text = (
        f"Normal Model Answer:\n"
        f"  -> '{case_info['nrm_answer']}' ({nrm_result})\n\n"
        f"{alt_model_name} Model Answer:\n"
        f"  -> '{case_info['alt_answer']}' ({alt_result})"
    )
    ax.text(0.05, 0.9, answers_text, transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.4))
    ax.axis('off')

    # --- Panel 3 & 4 can be used for more details if needed, otherwise hide them ---
    axes[1, 0].set_title('3. Shuffled Input (Text)', fontweight='bold', fontsize=12)
    axes[1, 0].text(0.05, 0.9, wrap_text(case_info.get('shuffled_text_q', 'N/A')), transform=axes[1, 0].transAxes, fontsize=10, va='top')
    axes[1, 0].axis('off')

    axes[1, 1].set_title('4. Shuffled Input (Image)', fontweight='bold', fontsize=12)
    shuffled_img = case_info.get('shuffled_image', None)
    if shuffled_img:
        axes[1, 1].imshow(shuffled_img)
    else:
        axes[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
    axes[1, 1].axis('off')

    # --- Finalize and Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join(category_dir, f"case_{qid}.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved analysis image: {output_filename}")

    # Save original image separately
    if case_info["image"] is not None:
        original_image_path = os.path.join(category_dir, f"case_{qid}_original.png")
        case_info["image"].save(original_image_path)
    
    # Save shuffled image if it exists
    shuffled_img = case_info.get('shuffled_image', None)
    if shuffled_img:
        shuffled_image_path = os.path.join(category_dir, f"case_{qid}_shuffled.png")
        shuffled_img.save(shuffled_image_path)

def save_sanity_check_plot(output_dir, dataset_name, condition, example_index, original_image, original_question, model_input_image, model_input_question, model_answer, gt_answer):
    """Saves a plot for a single sanity check example."""
    sanity_dir = os.path.join(output_dir, dataset_name, "sanity_checks", condition)
    os.makedirs(sanity_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Sanity Check: {condition.replace("_", " ").title()} - Example {example_index}', fontsize=16)

    # --- Panel 1: Original Correct Pair ---
    ax = axes[0, 0]
    if original_image:
        ax.imshow(original_image)
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
    ax.set_title('1. Original Correct Pair', fontweight='bold', fontsize=12)
    ax.axis('off')
    orig_text = f"Q: {wrap_text(original_question)}\n\nGround Truth: {gt_answer}"
    ax.text(0, -0.1, orig_text, transform=ax.transAxes, fontsize=10, va='top')

    # --- Panel 2: Model Input ---
    ax = axes[0, 1]
    if model_input_image:
        ax.imshow(model_input_image)
    else:
        ax.text(0.5, 0.5, 'No Image Input', ha='center', va='center', fontsize=14)
    ax.set_title('2. Actual Model Input', fontweight='bold', fontsize=12)
    ax.axis('off')
    
    # Show shuffle status
    is_img_shuffled = " [Shuffled]" if condition == "image_shuffle" else ""
    is_txt_shuffled = " [Shuffled]" if condition == "text_shuffle" else ""
    model_input_text = f"Image:{is_img_shuffled}\nQuestion:{is_txt_shuffled}\n\nQ: {wrap_text(model_input_question)}"
    ax.text(0, -0.1, model_input_text, transform=ax.transAxes, fontsize=10, va='top')

    # --- Panel 3: Model & GT Answers ---
    ax = axes[1, 0]
    ax.set_title('3. Model vs. Ground Truth', fontweight='bold', fontsize=12)
    options_text = (
        f"Implicit Options:\n"
        f"  - Yes\n"
        f"  - No\n\n"
        f"Model's Answer:\n"
        f"  -> '{model_answer}'\n\n"
        f"Correct Answer:\n"
        f"  -> '{gt_answer}'"
    )
    ax.text(0.05, 0.9, options_text, transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.4))
    ax.axis('off')

    # --- Panel 4: Final Evaluation ---
    ax = axes[1, 1]
    ax.set_title('4. Final Evaluation Result', fontweight='bold', fontsize=12)
    is_correct = model_answer.strip().lower() == gt_answer.strip().lower()
    result_text = 'CORRECT' if is_correct else 'INCORRECT'
    result_color = 'green' if is_correct else 'red'
    ax.text(0.5, 0.5, result_text, transform=ax.transAxes, fontsize=48, fontweight='bold',
            ha='center', va='center', color='white',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=result_color, alpha=0.9))
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join(sanity_dir, f"example_{example_index}.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_agreement_matrix_plot(normal_data, shuffle_data, shuffle_type, output_dir, model_name, dataset_name):
    """Creates a 2x2 agreement matrix between Normal and a Shuffled condition."""
    
    normal_answers = {d['question_id']: str(d['answer']).lower().strip() == str(d['gt_answer']).lower().strip() for d in normal_data}
    shuffle_answers = {d['question_id']: str(d['answer']).lower().strip() == str(d['gt_answer']).lower().strip() for d in shuffle_data}
    
    # [Normal Correct/Shuffle Correct, Normal Correct/Shuffle Incorrect]
    # [Normal Incorrect/Shuffle Correct, Normal Incorrect/Shuffle Incorrect]
    matrix = [[0, 0], [0, 0]]
    
    qids = set(normal_answers.keys()) & set(shuffle_answers.keys())
    if not qids:
        print(f"Warning: No common questions between Normal and {shuffle_type} to create agreement matrix.")
        return

    for qid in qids:
        normal_correct = normal_answers.get(qid, False)
        shuffle_correct = shuffle_answers.get(qid, False)
        if normal_correct and shuffle_correct:
            matrix[0][0] += 1
        elif normal_correct and not shuffle_correct:
            matrix[0][1] += 1
        elif not normal_correct and shuffle_correct:
            matrix[1][0] += 1
        else: # not normal_correct and not shuffle_correct
            matrix[1][1] += 1

    total = len(qids)
    if total == 0:
        print(f"Warning: Zero common questions, cannot generate agreement matrix for {shuffle_type}.")
        return
        
    matrix_percent = np.array(matrix) / total * 100
    
    labels = [f"{v}\n({p:.1f}%)" for v, p in zip(np.array(matrix).flatten(), matrix_percent.flatten())]
    labels = np.asarray(labels).reshape(2, 2)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix_percent, annot=labels, fmt='', cmap='Greens', cbar=False,
                xticklabels=[f'{shuffle_type}\nCorrect', f'{shuffle_type}\nIncorrect'],
                yticklabels=['Normal\nCorrect', 'Normal\nIncorrect'],
                annot_kws={"size": 16}, linewidths=.5, linecolor='black')
    
    plt.title(f'Correctness Agreement: Normal vs. {shuffle_type}\n{model_name.upper()} on {dataset_name.upper()}', fontsize=16)
    plt.ylabel('Normal Condition', fontsize=12)
    plt.xlabel(f'{shuffle_type} Condition', fontsize=12)
    plt.tight_layout()

    plot_dir = os.path.join(output_dir, dataset_name, "agreement_plots")
    os.makedirs(plot_dir, exist_ok=True)
    sanitized_shuffle_type = shuffle_type.replace(" ", "_")
    output_filename = os.path.join(plot_dir, f"agreement_matrix_{model_name}_{sanitized_shuffle_type}.pdf")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved agreement matrix to: {output_filename}")


def select_balanced_examples(cases, num_examples):
    """
    Selects a random, balanced sample of cases based on yes/no answers.
    Prioritizes balance, even if it means returning fewer than num_examples.
    """
    yes_cases = [c for c in cases if str(c['gt_answer']).strip().lower() == 'yes']
    no_cases = [c for c in cases if str(c['gt_answer']).strip().lower() == 'no']

    random.shuffle(yes_cases)
    random.shuffle(no_cases)

    num_per_category = num_examples // 2
    
    selected_yes = yes_cases[:min(len(yes_cases), num_per_category)]
    selected_no = no_cases[:min(len(no_cases), num_per_category)]
    
    if len(selected_yes) != len(selected_no):
        min_len = min(len(selected_yes), len(selected_no))
        selected_yes = selected_yes[:min_len]
        selected_no = selected_no[:min_len]

    selected_cases = selected_yes + selected_no
    
    remaining_needed = num_examples - len(selected_cases)
    if remaining_needed > 0:
        remaining_pool = [c for c in cases if c not in selected_cases]
        random.shuffle(remaining_pool)
        selected_cases.extend(remaining_pool[:remaining_needed])

    random.shuffle(selected_cases)
    return selected_cases

def analyze(args):
    # Load results from JSONL files
    try:
        nrm_results = {item['question_id']: item for item in load_jsonl(os.path.join(args.results_dir, 'ensemble_nrm.jsonl'))}
        txt_results = {item['question_id']: item for item in load_jsonl(os.path.join(args.results_dir, 'ensemble_txt.jsonl'))} # Image Shuffle (--image_shuffle)
        img_results = {item['question_id']: item for item in load_jsonl(os.path.join(args.results_dir, 'ensemble_img.jsonl'))} # Text Shuffle (--text_shuffle)
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e.filename}")
        print("Please ensure that 'ensemble_nrm.jsonl', 'ensemble_txt.jsonl', and 'ensemble_img.jsonl' exist in the specified directory.")
        return

    # Extract dataset name from the results directory path
    dataset_name = os.path.basename(os.path.normpath(args.results_dir))
    model_name = os.path.basename(os.path.normpath(os.path.dirname(args.results_dir))).replace('answers_', '')

    # --- SINGLE SOURCE OF TRUTH FOR SHUFFLING ---
    # Match the exact shuffling sequence from mme_eval.py
    from datasets import load_dataset
    questions_data = load_dataset("lmms-lab/MME", split="test", cache_dir=args.hf_cache_dir)
    
    # First shuffle creates the base order (this is 'line' in mme_eval.py)
    questions = list(questions_data.shuffle(seed=19))
    
    # These create the wrong_line1 and wrong_line2 that were used in mme_eval.py
    wrong_line1 = list(questions_data.shuffle(seed=42))  # For text shuffle
    wrong_line2 = list(questions_data.shuffle(seed=73))  # For image shuffle

    # --- Generate Agreement Matrices ---
    print("\n--- Generating Agreement Matrices ---")
    create_agreement_matrix_plot(
        list(nrm_results.values()), list(img_results.values()), 
        "Text Shuffle", args.output_dir, model_name, dataset_name
    )
    create_agreement_matrix_plot(
        list(nrm_results.values()), list(txt_results.values()),
        "Image Shuffle", args.output_dir, model_name, dataset_name
    )

    # --- Sanity Checks for first 3 examples ---
    print("\n--- Generating Sanity Check Plots for First 3 Examples ---")
    for i in range(3):
        qid = i  # In the eval script, the question_id was just the loop index
        
        # Get examples from each dataset in the same order as during evaluation
        base_example = questions[i]
        text_shuffle_example = wrong_line1[i]
        image_shuffle_example = wrong_line2[i]
        
        # Normal condition - uses base order
        if qid in nrm_results:
            save_sanity_check_plot(
                args.output_dir, dataset_name, "normal", i,
                base_example["image"], base_example["question"],  # Original pair
                base_example["image"], base_example["question"],  # Same for model input
                nrm_results[qid]['answer'], nrm_results[qid]['gt_answer']
            )

        # Text Shuffle (img_results from --text_shuffle)
        # Uses wrong_line1's question (seed 42) with original image
        if qid in img_results:
            save_sanity_check_plot(
                args.output_dir, dataset_name, "text_shuffle", i,
                base_example["image"], base_example["question"],  # Original pair
                base_example["image"], text_shuffle_example["question"],  # Model input: original image + shuffled text
                img_results[qid]['answer'], img_results[qid]['gt_answer']
            )

        # Image Shuffle (txt_results from --image_shuffle)
        # Uses wrong_line2's image (seed 73) with original question
        if qid in txt_results:
            save_sanity_check_plot(
                args.output_dir, dataset_name, "image_shuffle", i,
                base_example["image"], base_example["question"],  # Original pair
                image_shuffle_example["image"], base_example["question"],  # Model input: shuffled image + original text
                txt_results[qid]['answer'], txt_results[qid]['gt_answer']
            )
    
    # --- Analysis of Disagreements ---
    analysis_cases = defaultdict(list)

    # --- Compare Normal vs. Text Shuffle ---
    for qid, item in img_results.items():  # img_results has text shuffle results
        if qid not in nrm_results: continue
        
        nrm_item = nrm_results[qid]
        is_nrm_correct = nrm_item['answer'].strip().lower() == nrm_item['gt_answer'].strip().lower()
        is_alt_correct = item['answer'].strip().lower() == item['gt_answer'].strip().lower()

        # For text shuffle: use wrong_line1's question with original image
        original_line = questions[qid]
        wrong_line1_data = wrong_line1[qid]

        case_data = {
            "qid": qid, "category": item['category'],
            "image": original_line["image"],           # From original line
            "question": original_line["question"],
            "gt_answer": item['gt_answer'], 
            "nrm_answer": nrm_item['answer'],
            "alt_answer": item['answer'], 
            "shuffled_text_q": wrong_line1_data["question"],  # From wrong_line1
            "nrm_correct": is_nrm_correct, 
            "alt_correct": is_alt_correct
        }

        if is_alt_correct and not is_nrm_correct:
            analysis_cases['normal_incorrect_vs_text_shuffle_correct'].append(case_data)
        elif is_alt_correct and is_nrm_correct:
            analysis_cases['normal_correct_vs_text_shuffle_correct'].append(case_data)

    # --- Compare Normal vs. Image Shuffle ---
    for qid, item in txt_results.items():  # txt_results has image shuffle results
        if qid not in nrm_results: continue

        nrm_item = nrm_results[qid]
        is_nrm_correct = nrm_item['answer'].strip().lower() == nrm_item['gt_answer'].strip().lower()
        is_alt_correct = item['answer'].strip().lower() == item['gt_answer'].strip().lower()

        # For image shuffle: use wrong_line2's image with original question
        original_line = questions[qid]
        wrong_line2_data = wrong_line2[qid]

        case_data = {
            "qid": qid, "category": item['category'],
            "image": original_line["image"],
            "question": original_line["question"],      # From original line
            "gt_answer": item['gt_answer'], 
            "nrm_answer": nrm_item['answer'],
            "alt_answer": item['answer'], 
            "shuffled_image": wrong_line2_data["image"],  # From wrong_line2
            "nrm_correct": is_nrm_correct, 
            "alt_correct": is_alt_correct
        }

        if is_alt_correct and not is_nrm_correct:
            analysis_cases['normal_incorrect_vs_image_shuffle_correct'].append(case_data)
        elif is_alt_correct and is_nrm_correct:
            analysis_cases['normal_correct_vs_image_shuffle_correct'].append(case_data)
    
    # --- Save results for all collected cases ---
    for case_type, cases in analysis_cases.items():
        print(f"\nFound {len(cases)} cases for type: {case_type}")
        if not cases:
            continue
        
        selected_to_save = select_balanced_examples(cases, args.num_examples)
        
        print(f"Saving {len(selected_to_save)} randomly selected, balanced examples...")
        for case in selected_to_save:
            save_analysis_image(case, args.output_dir, dataset_name, case_type)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze disagreements between normal and shuffled evaluation results.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing the 'ensemble_*.jsonl' result files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the analysis images.")
    parser.add_argument('--hf_cache_dir', type=str, default='./hf_cache', help="Directory for Hugging Face cache.")
    parser.add_argument('--num_examples', type=int, default=10, help="Number of disagreement examples to save for each type.")
    args = parser.parse_args()
    
    analyze(args) 