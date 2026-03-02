import argparse
import os
import sys
import json
import random
import torch
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

import math

eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

cambrian_path = os.path.dirname(eval_dir)
if cambrian_path not in sys.path:
    sys.path.insert(0, cambrian_path)

from model_loader import load_model_by_type, detect_model_type


def split_list(lst, n):
    chunk_size = math.ceil(lst / n) 
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _select_image_with_shuffle(line, wrong_line2, do_image_shuffle):
    base_imgs = line.get("image", [])
    if base_imgs is None:
        base_imgs = []
    if len(base_imgs) == 0:
        return None

    chosen = base_imgs[0]
    if do_image_shuffle:
        cand_imgs = wrong_line2.get("image", [])
        if cand_imgs:
            chosen = cand_imgs[0]

    return chosen


def process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    qs = (wrong_line1["question"] if args.text_shuffle else line["question"]) + " Options:"
    qs += ("\nA. " + line["choice_a"])
    qs += ("\nB. " + line["choice_b"])
    qs += ("\nC. " + line["choice_c"])
    qs += ("\nD. " + line["choice_d"])
    qs += f"\n{args.question_extension}"

    chosen_image = _select_image_with_shuffle(line, wrong_line2, args.image_shuffle)
    input_image = None
    if chosen_image is not None:
        input_image = chosen_image.convert('RGB')

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
        image_size = [input_image.size]
        image_tensor = process_images([input_image], image_processor, model_config)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)
    return input_ids, image_tensor, image_size, prompt


def process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type):
    qs = (wrong_line1["question"] if args.text_shuffle else line["question"]) + " Options:"
    qs += ("\nA. " + line["choice_a"])
    qs += ("\nB. " + line["choice_b"])
    qs += ("\nC. " + line["choice_c"])
    qs += ("\nD. " + line["choice_d"])
    qs += f"\n{args.question_extension}"
    
    chosen_image = _select_image_with_shuffle(line, wrong_line2, args.image_shuffle)
    input_image = None
    if chosen_image is not None:
        input_image = chosen_image.convert('RGB')
    
    if model_type in ['qwen2_5', 'qwen3']:
        messages = [{"role": "user", "content": []}]
        if input_image is not None:
            messages[0]["content"].append({"type": "image", "image": input_image})
        messages[0]["content"].append({"type": "text", "text": qs})
        
        text = image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = image_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        inputs = inputs.to('cuda')
        return inputs, None, None, qs
    else:  
        if input_image is not None:
            prompt = f"<image>\n{qs}"
        else:
            prompt = qs
        
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
        return process_qwen_llava(
            line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type
        )
    else:  
        return process_cambrian(
            line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config
        )


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
    questions = load_dataset("lmms-lab/SEED-Bench", split="test")
    
    shuffle_questions1 = questions.shuffle(seed=42) if args.text_shuffle or args.image_shuffle else questions
    shuffle_questions2 = questions.shuffle(seed=73) if args.text_shuffle or args.image_shuffle else questions

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)

    for line, wrong_line1, wrong_line2 in tqdm(
        zip(questions, shuffle_questions1, shuffle_questions2),
        total=len(questions)
    ):
        idx += 1

        images = line.get('image', [])
        if images is None:
            images = []
        if len(images) == 0:
            continue  

        if idx < valid_chunk[0] or idx > valid_chunk[1]:
            continue

        inputs, image_tensor, image_sizes, prompt = process(
            line, wrong_line1, wrong_line2, args,
            tokenizer, image_processor, model.config, args.model_type
        )
        gt_answer = line["answer"]
        category = line["question_type_id"]
        
        
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

        ans_file.write(json.dumps({
            "question_id": idx,
            "answer": outputs,
            "gt_answer": gt_answer,
            "prompt": prompt,
            "category": category,
            "model_id": model_name
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="liuhaotian/llava-v1.5-7b"
    )
    parser.add_argument(
        "--model_type", type=str, default=None,
        choices=['qwen2_5', 'qwen3', 'llava-next', 'cambrian']
    )
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument(
        "--question_extension", type=str,
        default="Answer with the option's letter from the given choices directly."
    )
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text_shuffle", action='store_true', help="Enable text shuffle"
    )
    parser.add_argument(
        "--image_shuffle", action='store_true', help="Enable image shuffle"
    )
    args = parser.parse_args()

    eval_model(args)
