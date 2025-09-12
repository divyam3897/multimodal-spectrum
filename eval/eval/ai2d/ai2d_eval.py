import argparse
import torch
import numpy as np
import random
import os
import json
from tqdm import tqdm
import shortuuid

from datasets import load_dataset
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk out of n chunks from a list of a certain length."""
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    """
    Processes a single data point, applying text and image shuffling as specified.
    """
    # 1. Select the question source based on the text_shuffle flag
    # Use original `line` for options, but shuffled `wrong_line1` for the question text
    question_source = wrong_line1 if args.text_shuffle else line
    qs = question_source["question"]
    
    # Options always come from the original, unshuffled line
    keys = ["A", "B", "C", "D"]
    # Ensure options are available and handle cases where they might be missing
    if "options" in line and line["options"] is not None:
        for i in range(len(line["options"])):
            option = line["options"][i]
            key = keys[i]
            qs += f"\n{key}. {option}"
    qs += f"\n{args.question_extension}"

    # 2. Select the image source based on the image_shuffle flag
    # This determines which record's image will be used
    img_line = wrong_line2 if args.image_shuffle else line

    # 3. Prepare the prompt and image tensor
    if img_line.get("image") is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if img_line.get("image") is None:
        image_tensor = None
        image_size = None
    else:
        image = img_line["image"].convert('RGB')
        image_tensor = process_images([image], image_processor, model_config)
        image_size = [image.size]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare datasets
    questions = load_dataset("lmms-lab/ai2d", split="test")
    shuffle_questions1 = questions.shuffle(seed=42)
    shuffle_questions2 = questions.shuffle(seed=73)

    # Prepare output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    chunk_fname = f"{os.path.splitext(os.path.basename(answers_file))[0]}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(os.path.dirname(answers_file), chunk_fname)
    ans_file = open(chunk_file, "w")

    # Get the chunk of data to process for this job
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    
    # Main evaluation loop
    for idx, (line, wrong_line1, wrong_line2) in enumerate(tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions))):
        if not (valid_chunk[0] <= idx <= valid_chunk[1]):
            continue

        input_ids, image_tensor, image_sizes, prompt = process(
            line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
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
        
        # Ground truth answer always comes from the original `line`
        gt_answer_idx = line["answer"]
        gt_answer_text = "N/A"
        if "options" in line and line["options"] is not None and gt_answer_idx is not None:
            try:
                gt_answer_text = line["options"][int(gt_answer_idx)]
            except (ValueError, IndexError):
                gt_answer_text = "Invalid index"


        # FIX: Renamed keys to match the downstream processing script
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "text_shuffled": args.text_shuffle,
            "image_shuffled": args.image_shuffle,
            "answer": outputs,  # Renamed from "model_output"
            "gt_answer": gt_answer_idx, # Renamed from "ground_truth_answer_idx"
            "ground_truth_answer_text": gt_answer_text,
            "model_id": model_name
        }) + "\n")
        ans_file.flush()
        
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/ai2d_answers.jsonl")
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
