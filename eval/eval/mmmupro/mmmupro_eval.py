import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from datasets import load_dataset, concatenate_datasets
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# Suppress verbose tokenizer warnings
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    # get kth chunk out of n chunks cut from lst length
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    # qs = line["question"]
    qs = wrong_line1["question"] if args.text_shuffle else line["question"]
    # if line["image_2"] is not None:
    #     return None, None, None, None

    # if line["question_type"] == "multiple-choice":
    qs += " Options:"
    options = re.findall(r"'(.*?)'", line["options"])
    for i in range(len(options)):
        option = options[i]
        qs += f"\n{chr(ord('A')+i)}. {option}"
    qs += f"\n{args.question_extension}"
    # else:
    #     qs += f"\nAnswer the question using a single word or phrase."
    img_line = wrong_line2 if args.image_shuffle else line

    if img_line["image_1"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    # remove <image \d> tags
    qs = re.sub(r'<image \d+>', '', qs).strip()

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if img_line["image_1"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = img_line["image_1"].convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt, qs, image


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print(f"load_pretrained_model returned: {args.model_base}, {model}")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.verbose = False
    
    validation_dataset = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    dev_dataset = load_dataset("lmms-lab/MMMU", split="dev")
    # questions = concatenate_datasets([validation_dataset, dev_dataset])
    questions = concatenate_datasets([validation_dataset])
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    
    # If only one chunk, use the original filename; otherwise add chunk suffix
    if args.num_chunks == 1:
        chunk_file = answers_file
    else:
        chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
        chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    idx = -1
    print("Number of total questions", len(questions))
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print("Valid chunk", valid_chunk)

    example_num = 0
    
    with open(chunk_file, "w") as ans_file:
        questions = questions.shuffle(seed=19)            
        shuffle_questions1 = questions.shuffle(seed=42)     
        shuffle_questions2 = questions.shuffle(seed=73)

        # Get all categories for visualization
        all_categories = sorted(set([ex["id"].split('_')[1] for ex in questions]))
        print(f"Categories found: {all_categories}")
        
        def plot_category_dist(dataset, title, filename, category_order=None):
            from collections import Counter
            cats = [ex["id"].split('_')[1] for ex in dataset]
            counter = Counter(cats)
            
            if category_order is not None:
                ordered_categories = category_order
                ordered_counts = [counter.get(cat, 0) for cat in ordered_categories]
            else:
                ordered_categories = list(counter.keys())
                ordered_counts = list(counter.values())
            
            plt.figure(figsize=(14, 8))
            colors = plt.cm.Set3(range(len(ordered_categories)))  
            bars = plt.bar(ordered_categories, ordered_counts, color=colors, edgecolor='black', alpha=0.8)
            
            plt.title(title, fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Category', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, count in zip(bars, ordered_counts):
                height = bar.get_height()
                if height > 0:  
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.92)
            
            pdf_filename = filename.replace('.png', '.pdf')
            plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()

        plot_dir = os.path.dirname(answers_file)
        shuffle_suffix = f"_txt{'ON' if args.text_shuffle else 'OFF'}_img{'ON' if args.image_shuffle else 'OFF'}"
        
        if not args.image_shuffle and not args.text_shuffle:
            plot_category_dist(questions, f"Category Distribution (MMMU-Pro Dataset)", os.path.join(plot_dir, f"category_distribution{shuffle_suffix}.png"), all_categories)

        for line, wrong_line1, wrong_line2 in tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions)):
            idx = idx+1
            if idx<valid_chunk[0] or idx>valid_chunk[1]:
                continue
            
            input_ids, image_tensor, image_sizes, prompt, qs, image = process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config)
            if input_ids is None:
                continue
            gt_answer = line["answer"]
            category = line["id"].split('_')[1]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            
            # Create attention mask to avoid warnings
            attention_mask = torch.ones_like(input_ids)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Debug visualization for first few examples
            if example_num < 3:
                debug_dir = os.path.dirname(answers_file)
                shuffle_suffix = f"_txt{'ON' if args.text_shuffle else 'OFF'}_img{'ON' if args.image_shuffle else 'OFF'}"
                debug_prefix = os.path.join(debug_dir, f"example_{example_num}{shuffle_suffix}")
                
                # Adaptive layout based on enabled shuffles
                num_shuffles = sum([args.text_shuffle, args.image_shuffle])
                
            if example_num < 3:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                fig.suptitle(f'Debug Example {example_num} | Question ID: {idx} | Category: {category}', fontsize=18, fontweight='bold')

                # Helper function for text wrapping
                def wrap_text(text, width=45):
                    import textwrap
                    return '\n'.join(textwrap.wrap(text, width=width))

                # --- Panel 1: Original Correct Pair ---
                ax = axes[0, 0]
                if line["image_1"] is not None:
                    ax.imshow(line["image_1"].convert('RGB'))
                else:
                    ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
                ax.set_title('1. Original Correct Pair', fontweight='bold', fontsize=12)
                ax.axis('off')
                orig_text = f"Q: {wrap_text(line['question'])}\n\nGround Truth: {line['answer']}"
                ax.text(0, -0.1, orig_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')


                # --- Panel 2: Model Input ---
                ax = axes[0, 1]
                if image is not None:
                    ax.imshow(image)
                else:
                    ax.text(0.5, 0.5, 'No Image Input', ha='center', va='center', fontsize=14)
                ax.set_title('2. Actual Model Input', fontweight='bold', fontsize=12)
                ax.axis('off')
                is_img_shuffled = " [Shuffled]" if args.image_shuffle else ""
                is_txt_shuffled = " [Shuffled]" if args.text_shuffle else ""
                model_input_text = f"Image:{is_img_shuffled}\nQuestion:{is_txt_shuffled}\n\nQ: {wrap_text(qs)}"
                ax.text(0, -0.1, model_input_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')


                # --- Panel 3: Model & GT Answers ---
                ax = axes[1, 0]
                ax.set_title('3. Model vs. Ground Truth', fontweight='bold', fontsize=12)
                options_str = ""
                # Check for options and display them
                if "options" in line and line["options"]:
                    options = line["options"]
                    options_str = "\n\nOptions:\n" + "\n".join([f"  ({chr(ord('A')+i)}) {opt}" for i, opt in enumerate(options)])
                
                answer_text = (
                    f"Model's Answer:\n"
                    f"  -> '{outputs}'\n\n"
                    f"Correct Answer:\n"
                    f"  -> '{gt_answer}'"
                    f"{options_str}"
                )
                ax.text(0.05, 0.95, answer_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.4))
                ax.axis('off')


                # --- Panel 4: Final Evaluation ---
                ax = axes[1, 1]
                ax.set_title('4. Final Evaluation Result', fontweight='bold', fontsize=12)
                is_correct = outputs.strip().lower() == gt_answer.strip().lower()
                result_text = 'CORRECT' if is_correct else 'INCORRECT'
                result_color = 'green' if is_correct else 'red'
                ax.text(0.5, 0.5, result_text, transform=ax.transAxes, fontsize=48, fontweight='bold',
                        ha='center', va='center', color='white',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=result_color, alpha=0.9))
                ax.axis('off')


                # --- Finalize and Save ---
                plt.tight_layout(rect=[0, 0.05, 1, 0.96])
                debug_dir = os.path.dirname(answers_file)
                shuffle_suffix = f"_txt{'ON' if args.text_shuffle else 'OFF'}_img{'ON' if args.image_shuffle else 'OFF'}"
                debug_prefix = os.path.join(debug_dir, f"example_{example_num}{shuffle_suffix}")
                plt.savefig(f"{debug_prefix}_comprehensive.pdf", dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"Saved comprehensive debug visualization: {debug_prefix}_comprehensive.pdf")
                
                example_num += 1
            
            ans_file.write(json.dumps({
                "model_id":model_name,
                "question_id": idx,
                "prompt": prompt,
                "answer": outputs,
                "gt_answer": gt_answer,
                "category": category,
                # "type": line["question_type"]
            }) + "\n")
            
            ans_file.flush()

    print(f"=== EVALUATION COMPLETE ===")
    print(f"Saved results to: {chunk_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_shuffle", action='store_true', help="Enable text shuffle")
    parser.add_argument("--image_shuffle", action='store_true', help="Enable image shuffle")
    args = parser.parse_args()

    eval_model(args)

