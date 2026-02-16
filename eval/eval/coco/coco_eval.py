import argparse
import os
import sys
import json
import random
import re
import math

import torch
import numpy as np
from tqdm import tqdm
import shortuuid

from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from cambrian.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Path setup
# ---------------------------
eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

cambrian_path = os.path.dirname(eval_dir)
if cambrian_path not in sys.path:
    sys.path.insert(0, cambrian_path)

# Universal loader
from model_loader import load_model_by_type, detect_model_type


# ---------------------------
# Helpers for chunking
# ---------------------------

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# ---------------------------
# Text-processing helpers
# ---------------------------

def get_stem_from_prompt(prompt: str) -> str:
    """
    Extract just the question stem from a prompt string by stripping off the
    multiple-choice options.

    This uses a heuristic: we look for the first line that starts with something
    like "(A)", "A.", "A)" etc. and treat everything before that as the stem.

    If we don't find such a marker, we return the full prompt as the stem.
    """
    # Normalize line endings
    text = prompt

    # Look for a newline followed by something that looks like an option marker
    # e.g. "\n(A)", "\nA.", "\nA)"
    m = re.search(r'\n\s*(\(?[A-D]\)?[.\)])', text)
    if m:
        return text[:m.start()].strip()
    else:
        return text.strip()


def build_question_text(line, wrong_line1, args):
    """
    Build the final question text (without image tokens / chat template):

    - If text_shuffle is OFF:
        * Use stem from `line["prompt"]`
        * Use options from `line["choices"]`
    - If text_shuffle is ON:
        * Use stem from `wrong_line1["prompt"]`
        * Use options from `line["choices"]` (so options + labels stay aligned)
    - Finally, append `args.question_extension`.
    """
    # 1) Pick stem source
    if args.text_shuffle:
        stem_source = wrong_line1
    else:
        stem_source = line

    # 2) Extract stem from its prompt
    stem = get_stem_from_prompt(stem_source["prompt"])

    # 3) Attach options from the *current* line, so labels remain correct
    qs = stem
    if "choices" in line and line["choices"] is not None:
        for i, opt in enumerate(line["choices"]):
            letter = chr(ord("A") + i)
            qs += f"\n({letter}) {opt}"

    # 4) Instruction
    qs += f"\n{args.question_extension}"
    return qs


# ---------------------------
# Model-specific processing
# ---------------------------

def process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    # Build question text with possibly shuffled stem, but original options
    qs = build_question_text(line, wrong_line1, args)

    # Image selection (can be shuffled)
    img_line = wrong_line2 if args.image_shuffle else line
    input_image = img_line["image"]

    # Add image tokens to the text if there is an image
    if input_image is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # Wrap in conversation template
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Process image
    if input_image is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = input_image.convert("RGB")
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt, img_line


def process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type):
    # Build question text with possibly shuffled stem, but original options
    qs = build_question_text(line, wrong_line1, args)

    # Image selection
    img_line = wrong_line2 if args.image_shuffle else line
    input_image = img_line["image"]

    if model_type in ["qwen2_5", "qwen3"]:
        messages = [{"role": "user", "content": []}]
        if input_image is not None:
            messages[0]["content"].append({"type": "image", "image": input_image})
        messages[0]["content"].append({"type": "text", "text": qs})

        text = image_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = image_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        return inputs, None, None, qs, img_line

    else:  # llava-next
        if input_image is not None:
            prompt = f"<image>\n{qs}"
        else:
            prompt = qs

        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(prompt, list):
            prompt = prompt[0]

        if input_image is not None:
            inputs = image_processor(
                text=prompt,
                images=input_image,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to("cuda")
        return inputs, None, None, qs, img_line


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config, model_type):
    if model_type in ["qwen2_5", "qwen3", "llava-next"]:
        return process_qwen_llava(
            line,
            wrong_line1,
            wrong_line2,
            args,
            tokenizer,
            image_processor,
            model_type,
        )
    else:  # cambrian
        return process_cambrian(
            line,
            wrong_line1,
            wrong_line2,
            args,
            tokenizer,
            image_processor,
            model_config,
        )


# ---------------------------
# Main eval function
# ---------------------------

def eval_model(args):
    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Detect model type if needed
    if args.model_type is None:
        args.model_type = detect_model_type(args.model_path)
        print(f"Detected model type: {args.model_type}")

    # Load model
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor, context_len = load_model_by_type(
        model_path,
        args.model_type,
        args.model_base,
    )

    model = torch.compile(model, mode="max-autotune")

    if args.model_type in ["qwen2_5", "qwen3"]:
        model_name = f"qwen-vl-{os.path.basename(model_path)}"
    elif args.model_type == "llava-next":
        model_name = f"llava-next-{os.path.basename(model_path)}"
    else:
        model_name = get_model_name_from_path(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loaded {args.model_type} model: {model_name}")

    # Dataset
    questions = load_dataset("SaiCharithaAkula21/benchmark_coco_filtered", split="train")

    # Output file setup
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
    print("Valid chunk:", valid_chunk)

    # Shuffled datasets for text / image shuffling
    if args.text_shuffle or args.image_shuffle:
        shuffle_questions1 = questions.shuffle(seed=42)
        shuffle_questions2 = questions.shuffle(seed=73)
    else:
        shuffle_questions1 = questions
        shuffle_questions2 = questions

    # Main loop
    for line, wrong_line1, wrong_line2 in tqdm(
        zip(questions, shuffle_questions1, shuffle_questions2),
        total=len(questions),
    ):
        idx += 1
        if idx < valid_chunk[0] or idx > valid_chunk[1]:
            continue

        inputs, image_tensor, image_sizes, prompt, img_line = process(
            line,
            wrong_line1,
            wrong_line2,
            args,
            tokenizer,
            image_processor,
            model.config,
            args.model_type,
        )

        with torch.inference_mode():
            if args.model_type == "cambrian":
                inputs = inputs.to(device="cuda", non_blocking=True)
                attention_mask = torch.ones_like(inputs)
                output_ids = model.generate(
                    inputs,
                    images=image_tensor,
                    attention_mask=attention_mask,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                outputs = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                )[0].strip()
            else:
                # Qwen & LLaVA-NeXT
                input_len = inputs.input_ids.shape[1]
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                    if args.model_type in ["qwen2_5", "qwen3"]
                    else (True if args.temperature > 0 else False),
                    num_beams=1
                    if args.model_type in ["qwen2_5", "qwen3"]
                    else args.num_beams,
                    temperature=None
                    if args.model_type in ["qwen2_5", "qwen3"]
                    else (args.temperature if args.temperature > 0 else None),
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated_ids_trimmed = generated_ids[:, input_len:]
                decoder = (
                    image_processor
                    if args.model_type in ["qwen2_5", "qwen3"]
                    else tokenizer
                )
                outputs = decoder.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

        ans_file.write(
            json.dumps(
                {
                    "questionId": idx,
                    "image": img_line["img_name"],         # actual image used
                    "prompt": prompt,                      # full chat-formatted prompt
                    "answer": outputs,
                    "gt_answer": line["answer"],           # label from original line
                    "category": line["sub_task"],
                    "options": line["choices"],
                    "model_id": model_name,
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()
    print("Saved results to:", chunk_file)


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["qwen2_5", "qwen3", "llava-next", "cambrian"],
    )
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument(
        "--answers_file",
        type=str,
        default="./answers/answers.jsonl",
    )
    parser.add_argument(
        "--question_extension",
        type=str,
        default=(
            "Answer with the option's letter from the given choices directly. "
            "If you don't know the answer OR if you think that the correct answer "
            "is not present in the given options, select the option that says "
            "\"None of the above/I don't know\"."
        ),
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
        "--text_shuffle",
        action="store_true",
        help="Enable text shuffle (question stem only; options stay from original)",
    )
    parser.add_argument(
        "--image_shuffle",
        action="store_true",
        help="Enable image shuffle",
    )
    args = parser.parse_args()

    eval_model(args)
