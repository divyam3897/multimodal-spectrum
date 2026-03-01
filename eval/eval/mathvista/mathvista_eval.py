import argparse
import os
import sys
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt
import math

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from qwen_vl_utils import process_vision_info
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# Add paths
eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)
    
cambrian_path = os.path.dirname(eval_dir)
if cambrian_path not in sys.path:
    sys.path.insert(0, cambrian_path)

# Universal loader
from model_loader import load_model_by_type, detect_model_type


demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt

def extract_answer(model, response, problem, tokenizer, quick_extract=False, model_type='cambrian'):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    # Improved multiple choice handling
    if question_type == 'multi_choice':
        if response in choices:
            return response
        
        # Look for choice letters (A, B, C, D) - prioritize last occurrence
        choice_pattern = r'\b([A-Z])\b'
        choice_matches = re.findall(choice_pattern, response)
        if choice_matches:
            return choice_matches[-1]  # Return the last found choice letter
        
        # Look for "Answer is X" patterns
        answer_patterns = [
            r'Answer is ([A-Z])\.',
            r'Answer: ([A-Z])',
            r'answer is ([A-Z])',
            r'\(([A-Z])\)',
            r'The correct answer is \(([A-Z])\)',
            r'The answer is \(([A-Z])\)'
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # If we still haven't found it, look for "Answer:" pattern
        if "Answer:" in response:
            return response.replace("Answer:", "").strip()

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    # Quick extraction patterns
    if "(" in response and ")" in response and (response.index("(") == response.index(")") - 2):
        extraction = response[response.index("(") + 1]
        return extraction
    result = re.search(r'The answer is "(.*)"\.', response)
    if result:
        return result.group(1)
    result = re.search(r'Answer:"(.*)"\.', response)
    if result:
        return result.group(1).strip()
    
    # Use model for extraction (skip for QWEN)
    if model_type != 'qwen':
        full_prompt = create_test_prompt(demo_prompt, query, response)
        input_prompt = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_prompt)
        extraction = model.generate(
            input_prompt,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        extraction = tokenizer.batch_decode(extraction, skip_special_tokens=True)[0].strip()
        return extraction
    
    return ""


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    if args.text_shuffle:
        if line["question_type"] == "multi_choice":
            qs = wrong_line1["query"][: wrong_line1["query"].find("Choices") if wrong_line1["query"].find("Choices") != -1 else len(wrong_line1["query"])] + line["query"][line["query"].find("Choices"):]
        else:
            qs = wrong_line1["query"]
    else:
        qs = line["query"]

    img_line = wrong_line2 if args.image_shuffle else line
    
    # Add question extension
    if line["question_type"] == "multi_choice":
        qs += f"\n{args.question_extension}"
    else:
        qs += f"\nAnswer the question using a single word or phrase."
    
    if img_line["decoded_image"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if img_line["decoded_image"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = img_line["decoded_image"].convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    return input_ids, image_tensor, image_size, qs, image


def process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type):
    if args.text_shuffle:
        if line["question_type"] == "multi_choice":
            qs = wrong_line1["query"][: wrong_line1["query"].find("Choices") if wrong_line1["query"].find("Choices") != -1 else len(wrong_line1["query"])] + line["query"][line["query"].find("Choices"):]
        else:
            qs = wrong_line1["query"]
    else:
        qs = line["query"]

    img_line = wrong_line2 if args.image_shuffle else line
    
    # Add question extension
    if line["question_type"] == "multi_choice":
        qs += f"\n{args.question_extension}"
    else:
        qs += f"\nAnswer the question using a single word or phrase."
    
    if model_type in ['qwen2_5', 'qwen3']:
        messages = [{"role": "user", "content": []}]
        if img_line["decoded_image"] is not None:
            messages[0]["content"].append({"type": "image", "image": img_line["decoded_image"]})
        messages[0]["content"].append({"type": "text", "text": qs})
        
        text = image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = image_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        return inputs, None, None, qs, img_line["decoded_image"]
    else:  # llava-next
        if img_line["decoded_image"] is not None:
            prompt = f"<image>\n{qs}"
        else:
            prompt = qs
        
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, list):
            prompt = prompt[0]
        
        if img_line["decoded_image"] is not None:
            inputs = image_processor(text=prompt, images=img_line["decoded_image"], return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to('cuda')
        return inputs, None, None, qs, img_line["decoded_image"]


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

    # Detect model type if not provided
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
    tokenizer.verbose = False
    
    questions = load_dataset("AI4Math/MathVista", split="testmini")
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")
    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    existing_question_ids = set()
    if os.path.exists(chunk_file):
        with open(chunk_file, "r") as existing_file:
            for line in existing_file:
                line = line.strip()
                if not line or not line.startswith("{") or not line.endswith("}"):
                    continue
                record = json.loads(line)
                qid = record.get("question_id")
                if isinstance(qid, int):
                    existing_question_ids.add(qid)
        if existing_question_ids:
            print(f"Resuming from existing output: {len(existing_question_ids)} entries already present.")

    ans_file = open(chunk_file, "a")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    example_num = 0

    shuffle_questions1 = questions.shuffle(seed=42)
    shuffle_questions2 = questions.shuffle(seed=20)
    for line, wrong_line1, wrong_line2 in tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
        if idx in existing_question_ids:
            continue
        
        inputs, image_tensor, image_sizes, qs, image = process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config, args.model_type)
        if inputs is None:
            continue
        
        category = line["metadata"]["category"]
        task = line['metadata']["task"]
        gt_answer = line["answer"]
        if line["question_type"] == "multi_choice":
            reverse_dict = {}
            for ind, item in enumerate(line["choices"]):
                reverse_dict[item] = chr(ord('A')+ind)
            gt_answer = reverse_dict[gt_answer]
        

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
                    # Qwen3 models eference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
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

        real_output = extract_answer(model, outputs, line, tokenizer, quick_extract=False, model_type=args.model_type)

        ans_file.write(json.dumps({"model_id":model_name,
                                   "question_id": idx,
                                   "question": qs,
                                   "answer": real_output,
                                   "gt_answer": gt_answer,
                                   "category": category,
                                   "task": task,
                                   "type": line["question_type"]}) + "\n")
        example_num += 1
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_type", type=str, default=None, choices=['qwen2_5', 'qwen3', 'llava-next', 'cambrian'])
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="First show your reasoning process and then give the final answer.")
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
    parser.add_argument("--compile_model", action='store_true', help="Use torch.compile for faster inference (PyTorch 2.0+)")
    args = parser.parse_args()

    eval_model(args)

