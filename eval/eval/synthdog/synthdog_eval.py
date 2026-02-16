import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import json
import numpy as np
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets import load_dataset
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from qwen_vl_utils import process_vision_info
from model_loader import load_model_by_type, detect_model_type


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    """Processes a data sample for Cambrian models with independent text/image shuffling."""
    qs = "Please transcribe the text from the image word by word. Only include the words found in the image, and avoid adding any additional context or information."
    
    img_line = wrong_line2 if args.image_shuffle else line
    if img_line["image"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    img_line = wrong_line2 if args.image_shuffle else line
    if img_line["image"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = img_line["image"].convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size


def process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor):
    """Processes a data sample for Qwen and LLaVA-NeXT models with independent text/image shuffling."""
    qs = "Please transcribe the text from the image word by word. Only include the words found in the image, and avoid adding any additional context or information."
    
    img_line = wrong_line2 if args.image_shuffle else line
    
    if img_line["image"] is not None:
        image = img_line["image"].convert('RGB')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": qs}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": qs}
                ]
            }
        ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    return input_ids, image_inputs, None


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config, model_type):
    """Dispatcher function that calls the appropriate process function based on model type."""
    if model_type == 'cambrian':
        return process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config)
    else:  # qwen or llava-next
        return process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor)


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(args.model_path)
    
    # Detect model type if not provided
    if args.model_type is None:
        model_type = detect_model_type(model_path)
        print(f"Detected model type: {model_type}")
    else:
        model_type = args.model_type
    
    # Load model using universal loader
    tokenizer, model, image_processor, context_len = load_model_by_type(
        model_path=model_path,
        model_type=model_type,
        model_base=args.model_base
    )
    
    # Compile model for better performance
    if model_type == 'cambrian':
        model = torch.compile(model)
    
    # Set model_name based on model_type
    if model_type == 'cambrian':
        model_name = get_model_name_from_path(model_path)
    else:
        model_name = model_type
    
    # Adjust tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    questions = load_dataset("naver-clova-ix/synthdog-en", split="validation")
    
    
    # Create shuffled datasets for text and image shuffling
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
    # print(valid_chunk)
    for line, wrong_line1, wrong_line2 in tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
        
        input_ids, image_tensor, image_sizes = process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config, model_type)
        data = json.loads(line["ground_truth"])
        gt_answer = data['gt_parse']['text_sequence']

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        with torch.inference_mode():
            if model_type == 'cambrian':
                attention_mask = torch.ones_like(input_ids)
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
                    pad_token_id=tokenizer.pad_token_id)
            else:  # qwen or llava-next
                if image_tensor:
                    image_tensor = [img.to('cuda', dtype=torch.float16) for img in image_tensor]
                    inputs = {
                        'input_ids': input_ids,
                        'pixel_values': image_tensor[0] if len(image_tensor) > 0 else None,
                    }
                else:
                    inputs = {'input_ids': input_ids}
                
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "answer": outputs,
                                   "gt_answer": gt_answer,
                                   "model_id": model_name}) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None, choices=['qwen', 'llava-next', 'cambrian'],
                        help="Model type (qwen, llava-next, or cambrian). If not provided, will be detected automatically.")
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Answer the question using a single word or phrase.")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_shuffle", action='store_true', help="Enable text shuffle")
    parser.add_argument("--image_shuffle", action='store_true', help="Enable image shuffle")
    args = parser.parse_args()

    eval_model(args)

