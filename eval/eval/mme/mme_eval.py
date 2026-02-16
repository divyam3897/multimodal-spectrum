import argparse
import os
import sys
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt
import time
import string

from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

import math
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Add paths
eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

cambrian_path = os.path.dirname(eval_dir)
if cambrian_path not in sys.path:
    sys.path.insert(0, cambrian_path)

# Universal loader
from model_loader import load_model_by_type, detect_model_type


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    qs = wrong_line1["question"] if args.text_shuffle else line["question"]
    qs += f"\n{args.question_extension}"

    img_line = wrong_line2 if args.image_shuffle else line
    input_image = img_line["image"]
    
    if input_image is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if input_image is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = input_image.convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    return input_ids, image_tensor, image_size, prompt, image, qs


def process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type):
    qs = wrong_line1["question"] if args.text_shuffle else line["question"]
    qs += f"\n{args.question_extension}"
    
    img_line = wrong_line2 if args.image_shuffle else line
    input_image = img_line["image"]
    
    if model_type in ['qwen2_5', 'qwen3']:
        messages = [{"role": "user", "content": []}]
        if input_image is not None:
            messages[0]["content"].append({"type": "image", "image": input_image})
        messages[0]["content"].append({"type": "text", "text": qs})
        
        text = image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = image_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        return inputs, None, None, qs, input_image, qs
    else:  # llava-next
        if input_image is not None:
            prompt = f"<image>\n{qs}"
        else:
            prompt = qs
        
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, list):
            prompt = prompt[0]
        
        if input_image is not None:
            inputs = image_processor(text=prompt, images=input_image, return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to('cuda')
        return inputs, None, None, qs, input_image, qs


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config, model_type):
    if model_type in ['qwen2_5', 'qwen3', 'llava-next']:
        return process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type)
    else:  # cambrian
        return process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config)


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model_type is None:
        args.model_type = detect_model_type(args.model_path)
        print(f"Detected model type: {args.model_type}")

    # Load model using universal loader
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor, context_len = load_model_by_type(
        model_path, args.model_type, args.model_base
    )
    
    model = torch.compile(model, mode='max-autotune')
    
    if args.model_type in ['qwen2_5', 'qwen3']:
        model_name = f"qwen-vl-{os.path.basename(model_path)}"
    elif args.model_type == 'llava-next':
        model_name = f"llava-next-{os.path.basename(model_path)}"
    else:
        model_name = get_model_name_from_path(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loaded {args.model_type} model: {model_name}")

    questions = load_dataset("lmms-lab/MME", split="test", cache_dir="./hf_cache", keep_in_memory=True)

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
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
    
    all_categories = sorted(set([ex["category"] for ex in questions]))
    print(f"Categories found: {all_categories}")
    
    with open(chunk_file, "w") as ans_file:
        questions = questions.shuffle(seed=19)            
        shuffle_questions1 = questions.shuffle(seed=42)     
        shuffle_questions2 = questions.shuffle(seed=73)

        # Debugging and visualization
        print("Type of questions:", type(questions))
        print("Type of shuffle_questions1:", type(shuffle_questions1))
        print("Type of shuffle_questions2:", type(shuffle_questions2))
        print("Length of questions:", len(questions))
        print("Length of shuffle_questions1:", len(shuffle_questions1))
        print("Length of shuffle_questions2:", len(shuffle_questions2))

        # Print first 3 entries from each
        print("\nSample from questions:")
        for i in range(3):
            print(f"[{i}]", questions[i]["question"], "| category:", questions[i]["category"])
        print("\nSample from shuffle_questions1:")
        for i in range(3):
            print(f"[{i}]", shuffle_questions1[i]["question"], "| category:", shuffle_questions1[i]["category"])
        print("\nSample from shuffle_questions2:")
        for i in range(3):
            print(f"[{i}]", shuffle_questions2[i]["question"], "| category:", shuffle_questions2[i]["category"])

        def plot_category_dist(dataset, title, filename, category_order=None):
            from collections import Counter
            cats = [ex["category"] for ex in dataset]
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
            plot_category_dist(questions, f"Category Distribution (MME Dataset)", os.path.join(plot_dir, f"category_distribution{shuffle_suffix}.png"), all_categories)

        for line, wrong_line1, wrong_line2 in tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions)):
            idx = idx+1
            if idx<valid_chunk[0] or idx>valid_chunk[1]:
                continue

            inputs, image_tensor, image_sizes, prompt, img, qs = process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config, args.model_type)
            gt_answer = line["answer"]
            category = line["category"]
            
            
        with torch.inference_mode():
            if args.model_type == 'cambrian':
                    # Cambrian generation
                inputs = inputs.to(device='cuda', non_blocking=True)
                attention_mask = torch.ones_like(inputs)
                output_ids = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            else:
                input_len = inputs.input_ids.shape[1]
                if args.model_type == 'qwen3':
                    # Qwen3 models eference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
                    # greedy=false, top_p=0.8, top_k=20, temperature=0.7, repetition_penalty=1.0
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,  
                        temperature=0.7,  
                        top_p=0.8,  
                        top_k=20,  
                        repetition_penalty=1.0,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                elif args.model_type == 'qwen2_5':
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        temperature=None,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id
                        )
                else:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True if args.temperature > 0 else False,
                        num_beams=args.num_beams,
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                generated_ids_trimmed = generated_ids[:, input_len:]
                decoder = image_processor if args.model_type in ['qwen2_5', 'qwen3'] else tokenizer
                outputs = decoder.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if example_num < 3:
                debug_dir = os.path.dirname(answers_file)
                shuffle_suffix = f"_txt{'ON' if args.text_shuffle else 'OFF'}_img{'ON' if args.image_shuffle else 'OFF'}"
                debug_prefix = os.path.join(debug_dir, f"example_{example_num}{shuffle_suffix}")
                
                num_shuffles = sum([args.text_shuffle, args.image_shuffle])
         
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                translator = str.maketrans('', '', string.punctuation)
                outputs = outputs.strip().lower().translate(translator)
                gt_answer = gt_answer.strip().lower().translate(translator)
                
                # Added the question ID to the main title
                fig.suptitle(f'Debug Example {example_num} | Question ID: {idx} | Category: {category}', fontsize=18, fontweight='bold')

                # Helper function for text wrapping
                def wrap_text(text, width=45):
                    import textwrap
                    return '\n'.join(textwrap.wrap(text, width=width))

                # --- Panel 1: Original Correct Pair ---
                ax = axes[0, 0]
                if line["image"] is not None:
                    ax.imshow(line["image"])
                else:
                    ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
                ax.set_title('1. Original Correct Pair', fontweight='bold', fontsize=12)
                ax.axis('off')
                # Add original question and answer below the image
                orig_text = f"Q: {wrap_text(line['question'])}\n\nGround Truth: {line['answer']}"
                ax.text(0, -0.1, orig_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')


                ax = axes[0, 1]
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'No Image Input', ha='center', va='center', fontsize=14)
                ax.set_title('2. Actual Model Input', fontweight='bold', fontsize=12)
                ax.axis('off')
                # Determine shuffle status for the title text
                is_img_shuffled = " [Shuffled]" if args.image_shuffle else ""
                is_txt_shuffled = " [Shuffled]" if args.text_shuffle else ""
                # Add model input question below the image
                model_input_text = f"Image:{is_img_shuffled}\nQuestion:{is_txt_shuffled}\n\nQ: {wrap_text(qs.replace(args.question_extension, '').strip())}"
                ax.text(0, -0.1, model_input_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')


                # --- Panel 3: Model & GT Answers ---
                ax = axes[1, 0]
                ax.set_title('3. Model vs. Ground Truth', fontweight='bold', fontsize=12)
                options_text = (
                    f"Implicit Options:\n"
                    f"  - Yes\n"
                    f"  - No\n\n"
                    f"Model's Answer:\n"
                    f"  -> '{outputs}'\n\n"
                    f"Correct Answer:\n"
                    f"  -> '{gt_answer}'"
                )
                ax.text(0.05, 0.9, options_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
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
                plt.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust layout to prevent title overlap
                debug_dir = os.path.dirname(answers_file)
                shuffle_suffix = f"_txt{'ON' if args.text_shuffle else 'OFF'}_img{'ON' if args.image_shuffle else 'OFF'}"
                debug_prefix = os.path.join(debug_dir, f"example_{example_num}{shuffle_suffix}")
                plt.savefig(f"{debug_prefix}_comprehensive.pdf", dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"Saved comprehensive debug visualization: {debug_prefix}_comprehensive.pdf")
                
                
                # Save individual images separately for aper
                if line["image"] is not None and not args.image_shuffle and not args.text_shuffle:
                    plt.figure(figsize=(6, 6))
                    plt.imshow(line["image"])
                    plt.axis('off')
                    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
                    plt.savefig(f"{debug_prefix}_original_image.pdf", dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none', pad_inches=0)
                    plt.close()
                
                if img is not None and args.image_shuffle:
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
                    plt.savefig(f"{debug_prefix}_model_input_image.pdf", dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none', pad_inches=0)
                    plt.close()
                
                
                print(f"Saved comprehensive debug visualization: {debug_prefix}_comprehensive.pdf")
                example_num += 1

            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": prompt,
                "answer": outputs,
                "gt_answer": gt_answer,
                "category": category,
                "model_id": model_name
            }) + "\n")
            ans_file.flush()

    print(f"=== EVALUATION COMPLETE ===")
    print(f"Saved results to: {chunk_file}")
    print(f"Use mme_test.py to compute metrics from the saved results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_type", type=str, default=None, choices=['qwen2_5', 'qwen3', 'llava-next', 'cambrian'])
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Answer the question using a single word or phrase.")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_shuffle", action='store_true', help="Enable text shuffle")
    parser.add_argument("--image_shuffle", action='store_true', help="Enable image shuffle")
    args = parser.parse_args()

    eval_model(args)
