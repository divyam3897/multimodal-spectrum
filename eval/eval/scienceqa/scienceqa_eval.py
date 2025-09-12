import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    """
    Correctly processes a single data point by manually selecting components
    based on the shuffle flags.
    """
    # 1. Select the question source
    question_source = wrong_line1 if args.text_shuffle else line
    qs = ""
    if question_source["hint"] != "":
        qs = question_source["hint"] + " " + question_source["question"]
    else:
        qs = question_source["question"]
    
    # Options always come from the original `line` for correct evaluation
    for i in range(len(line["choices"])):
        option = line["choices"][i]
        qs += f"\n{chr(ord('A')+i)}. {option}"
    qs += f"\n{args.question_extension}"

    # 2. Select the image source
    img_line = wrong_line2 if args.image_shuffle else line

    # 3. Prepare the prompt and image
    image = None
    if img_line.get("image"):
        image = img_line["image"].convert('RGB')
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if image is None:
        image_tensor = None
        image_size = None
    else:
        image_tensor = process_images([image], image_processor, model_config)
        image_size = [image.size]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

    return input_ids, image_tensor, image_size, prompt, (image is not None)


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
    model = model.to(device='cuda')

    # Load and prepare datasets
    questions = load_dataset("derek-thomas/ScienceQA", split="test")
    shuffle_questions1 = questions.shuffle(seed=42)
    shuffle_questions2 = questions.shuffle(seed=73)

    # Prepare output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    chunk_fname = f"{os.path.splitext(os.path.basename(answers_file))[0]}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(os.path.dirname(answers_file), chunk_fname)
    ans_file = open(chunk_file, "w")

    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    
    # Main evaluation loop
    for idx, (line, wrong_line1, wrong_line2) in enumerate(tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions))):
        if not (valid_chunk[0] <= idx <= valid_chunk[1]):
            continue

        input_ids, image_tensor, image_sizes, prompt, is_multimodal = process(
            line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config
        )
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        if not is_multimodal:
             # Skip inference for text-only questions if model requires images
             # Or handle as a text-only task if your model supports it
            outputs = "SKIPPED_TEXT_ONLY"
        else:
            with torch.inference_mode():
                # Ensure images are a list of 4D tensors on CUDA (B, C, H, W), as expected by Cambrian
                if image_tensor is not None:
                    if isinstance(image_tensor, list):
                        processed_list = []
                        for t in image_tensor:
                            if t is None:
                                continue
                            if t.dim() == 3:
                                t = t.unsqueeze(0)
                            processed_list.append(t.to(device='cuda', non_blocking=True))
                        image_tensor = processed_list
                    else:
                        # Fallback: single tensor -> wrap as list
                        t = image_tensor
                        if t.dim() == 3:
                            t = t.unsqueeze(0)
                        image_tensor = [t.to(device='cuda', non_blocking=True)]
                
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
                    pad_token_id=tokenizer.pad_token_id,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Ground truth answer always comes from the original `line`
        gt_answer = line["answer"]

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "text_shuffled": args.text_shuffle,
            "image_shuffled": args.image_shuffle,
            "model_output": outputs,
            "ground_truth_answer": chr(ord('A') + gt_answer),
            "ground_truth_text": line["choices"][gt_answer],
            "model_id": model_name,
            "category": line["grade"]
        }) + "\n")
        ans_file.flush()
        
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/scienceqa_answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Please answer directly with only the letter of the correct option and nothing else.")
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