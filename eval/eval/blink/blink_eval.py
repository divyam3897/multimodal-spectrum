import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid

from datasets import load_dataset, concatenate_datasets
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    """Get kth chunk out of n chunks cut from lst length"""
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    text_source = wrong_line1 if args.text_shuffle else line
    image_source = wrong_line2 if args.image_shuffle else line

    qs = text_source["prompt"] + f"\n{args.question_extension}"

    image_data = image_source["image_1"]

    if image_data is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if image_data is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = image_data.convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.verbose = False

    all_questions_list = []
    categories = ["Counting", "IQ_Test", "Object_Localization", "Relative_Depth", "Relative_Reflectance", "Spatial_Relation"]
    print("Loading data from all categories...")
    for cat in tqdm(categories):
        dataset = load_dataset("BLINK-Benchmark/BLINK", cat, split="val")
        dataset = dataset.map(lambda example: {'category_name': cat, **example})
        all_questions_list.append(dataset)
    
    questions = concatenate_datasets(all_questions_list)
    questions = questions.shuffle(seed=42)
    print(f"Total questions loaded: {len(questions)}")

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
    
    shuffle_questions1 = questions.shuffle(seed=42)
    shuffle_questions2 = questions.shuffle(seed=73)
    
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    
    idx = -1
    for line, wrong_line1, wrong_line2 in tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions)):
        idx += 1
        if idx < valid_chunk[0] or idx > valid_chunk[1]:
            continue
            
        input_ids, image_tensor, image_sizes, prompt = process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config)
        gt_answer = line["answer"]
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "category": line["category_name"], 
            "model_id": model_name,
            "choices": line["choices"]
        }) + "\n")
        ans_file.flush()
    
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Please answer directly with only the letter of the correct option and nothing else.")
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