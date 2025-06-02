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

from datasets import load_dataset, concatenate_datasets
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
# from eval.eval.mathvista.ext_ans import demo_prompt
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader


from PIL import Image
import math


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

def extract_answer(model, response, problem, tokenizer, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""


    if question_type == 'multi_choice':
        if response in choices:
            return response
        elif "Answer:" in response:
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

    # quick extraction
    # if quick_extract:
    #     print("Quickly extracting answer...")
    #     # The answer is "text". -> "text"
    try:
        if (response.index("(") == response.index(")") - 2):
            extraction = response[response.index("(") + 1]
            return extraction
        result = re.search(r'The answer is "(.*)"\.', response)
        if result:
            extraction = result.group(1)
            return extraction
        result = re.search(r'Answer:"(.*)"\.', response)
        if result:
            extraction = result.group(1)
            return extraction.strip()
        full_prompt = create_test_prompt(demo_prompt, query, response)
        print("The full prompt inputed is:", full_prompt)
        input_prompt = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        extraction = model.generate(input_prompt)
        # real_output = extract_answer(model, extraction, line, tokenizer, quick_extract=True)
        extraction = tokenizer.batch_decode(extraction, skip_special_tokens=True)[0].strip()
        print("The extraction we decoded is ", extraction)
        return extraction
    except Exception as e:
        print(f"Error in extracting answer for problem: {pid} with response: {response}")
        print(e)
    return ""

    # # general extraction
    # try:
    #     full_prompt = create_test_prompt(demo_prompt, query, response)
    #     input_prompt = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    #     extraction = model.generate(input_prompt)
    #     # real_output = extract_answer(model, extraction, line, tokenizer, quick_extract=True)
    #     extraction = tokenizer.batch_decode(extraction, skip_special_tokens=True)[0].strip()
    #     print(extraction)
    #     return extraction
    # except Exception as e:
    #     print(f"Error in extracting answer for problem: {pid} with response: {response}")
    #     print(e)

    return ""


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    # get kth chunk out of n chunks cut from lst length
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, args, tokenizer, image_processor, model_config):
    qs = line["query"]

    if line["decoded_image"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # if line["question_type"] == "multi_choice":
    #     qs += f"\n{args.question_extension}"
    # else:
    #     qs += f"\nAnswer the question using a single word or phrase."
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if line["decoded_image"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = line["decoded_image"].convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, qs, image


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

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    example_num = 0
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
        
        input_ids, image_tensor, image_sizes, qs, image = process(line, args, tokenizer, image_processor, model.config)
        if input_ids is None:
            continue
        
        category = line["metadata"]["category"]
        task     = line['metadata']["task"]
        gt_answer = line["answer"]
        if line["question_type"] == "multi_choice":
            reverse_dict = {}
            for ind, item in enumerate(line["choices"]):
                reverse_dict[item] = chr(ord('A')+ind)
            gt_answer = reverse_dict[gt_answer]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        real_output = extract_answer(model, outputs, line, tokenizer, quick_extract=False)
        # extraction = tokenizer.batch_decode(real_output, skip_special_tokens=True)[0].strip()
        # print("Original Answer:", output_ids.cpu().numpy())
        print("Question:", line["query"])
        print("Direct from Model:", outputs)
        print("Extracted Answer:", real_output)
        # print("Extracted Answer3:", extraction)
        print("True Answer:", gt_answer)
        print("------------------------")
        ans_file.write(json.dumps({"model_id":model_name,
                                   "question_id": idx,
                                #    "answer_orig_np": output_ids.cpu().numpy(),
                                #    "answer_orig_np[]": output_ids.cpu().numpy().tolist(),
                                   "answer": real_output,
                                   "gt_answer": gt_answer,
                                   "category": category,
                                   "task": task,
                                   "type": line["question_type"]}) + "\n")
        print("example num:", example_num)
        if example_num < 3:
            try:
                fig, ax = plt.subplots(figsize=(6, 8))

                ax.axis("off")
                ax.imshow(image)

                caption_text = "Text:", qs, "\nY_Hat:", real_output,"\nTrue_Y:",gt_answer
                plt.figtext(0.5, 0.02, caption_text, wrap=True, horizontalalignment='center', fontsize=12)

                img_dir = "examples/runon/mathvista/norm_" + str(example_num) + ".png"
                plt.savefig(img_dir, bbox_inches='tight', dpi=300)

            except Exception as e:
                example_num -= 1
        else:
            print("all is done")
            return
        example_num += 1
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
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
    args = parser.parse_args()

    eval_model(args)

