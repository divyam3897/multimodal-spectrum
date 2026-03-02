import argparse
import os
import sys
import torch
import numpy as np
import random
import json
from tqdm import tqdm
import shortuuid

from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import math

eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

cambrian_path = os.path.dirname(eval_dir)
if cambrian_path not in sys.path:
    sys.path.insert(0, cambrian_path)

from model_loader import load_model_by_type, detect_model_type


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk out of n chunks from a list of a certain length."""
    chunks = split_list(lst, n)
    return chunks[k]


def process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    """
    Processes a single data point for Cambrian models, applying text and image shuffling as specified.
    """
    question_source = wrong_line1 if args.text_shuffle else line
    qs = question_source["question"]
    
    keys = ["A", "B", "C", "D"]
    if "options" in line and line["options"] is not None:
        for i in range(len(line["options"])):
            option = line["options"][i]
            key = keys[i]
            qs += f"\n{key}. {option}"
    qs += f"\n{args.question_extension}"

    img_line = wrong_line2 if args.image_shuffle else line

    input_image = img_line.get("image")
    
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
        image_tensor = None
        image_size = None
    else:
        image = input_image.convert('RGB')
        image_tensor = process_images([image], image_processor, model_config)
        image_size = [image.size]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    return input_ids, image_tensor, image_size, prompt


def process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type):
    """
    Processes a single data point for Qwen and LLaVA-NeXT models.
    """
    question_source = wrong_line1 if args.text_shuffle else line
    qs = question_source["question"]
    
    keys = ["A", "B", "C", "D"]
    if "options" in line and line["options"] is not None:
        for i in range(len(line["options"])):
            option = line["options"][i]
            key = keys[i]
            qs += f"\n{key}. {option}"
    qs += f"\n{args.question_extension}"

    img_line = wrong_line2 if args.image_shuffle else line
    input_image = img_line.get("image")
    
    if model_type in ['qwen2_5', 'qwen3']:
        messages = [{"role": "user", "content": []}]
        if input_image is not None:
            messages[0]["content"].append({"type": "image", "image": input_image})
        messages[0]["content"].append({"type": "text", "text": qs})
        
        text = image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = image_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        return inputs, None, None, qs
    else:  
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
        return inputs, None, None, qs


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config, model_type):
    if model_type in ['qwen2_5', 'qwen3', 'llava-next']:
        return process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type)
    else:  
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
    
    questions = load_dataset("lmms-lab/ai2d", split="test")
    shuffle_questions1 = questions.shuffle(seed=42)
    shuffle_questions2 = questions.shuffle(seed=73)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    chunk_fname = f"{os.path.splitext(os.path.basename(answers_file))[0]}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(os.path.dirname(answers_file), chunk_fname)
    ans_file = open(chunk_file, "w")

    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    
    for idx, (line, wrong_line1, wrong_line2) in enumerate(tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions))):
        if not (valid_chunk[0] <= idx <= valid_chunk[1]):
            continue

        inputs, image_tensor, image_sizes, prompt = process(
            line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config, args.model_type
        )


        with torch.inference_mode():
            if args.model_type == 'cambrian':
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

        gt_answer_idx = line["answer"]
        gt_answer_text = "N/A"
        if "options" in line and line["options"] is not None and gt_answer_idx is not None:
            try:
                gt_answer_text = line["options"][int(gt_answer_idx)]
            except (ValueError, IndexError):
                gt_answer_text = "Invalid index"

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "text_shuffled": args.text_shuffle,
            "image_shuffled": args.image_shuffle,
            "answer": outputs,  
            "gt_answer": gt_answer_idx, 
            "ground_truth_answer_text": gt_answer_text,
            "model_id": model_name
        }) + "\n")
        ans_file.flush()
        
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_type", type=str, default=None, choices=['qwen2_5', 'qwen3', 'llava-next', 'cambrian'])
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
