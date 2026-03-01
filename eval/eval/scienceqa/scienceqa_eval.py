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
from cambrian.conversation import conv_templates
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import math

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
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _select_image_with_shuffle(line, wrong_line2, do_image_shuffle):
    """
    Decide which image to actually feed to the model, and whether the *original*
    sample is multimodal.

    - Original multimodal property is determined by line["image"].
    - If image_shuffle is enabled, we try to use wrong_line2["image"].
      If that is None, we fall back to the original line["image"].
    """
    base_image = line.get("image", None)
    shuffled_image = wrong_line2.get("image", None) if do_image_shuffle else None

    # Original multimodal flag: did the original example have an image?
    is_multimodal = base_image is not None

    # Chosen image for the model:
    if do_image_shuffle:
        # Prefer shuffled image if it exists; otherwise fall back to original
        chosen_image = shuffled_image if shuffled_image is not None else base_image
    else:
        chosen_image = base_image

    return chosen_image, is_multimodal


def process_cambrian(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    """
    Processes a data sample for Cambrian models with independent text/image shuffling,
    and safe image fallback.
    """
    # Text shuffling
    question_source = wrong_line1 if args.text_shuffle else line

    # Build question string with hint + choices
    hint = question_source.get("hint")
    if hint:
        qs = hint + " " + question_source["question"]
    else:
        qs = question_source["question"]

    for i, option in enumerate(line["choices"]):
        qs += f"\n{chr(ord('A') + i)}. {option}"
    qs += f"\n{args.question_extension}"

    # Image selection with safe fallback
    chosen_image, is_multimodal = _select_image_with_shuffle(
        line, wrong_line2, args.image_shuffle
    )

    input_image = None
    if chosen_image is not None:
        # ScienceQA images may already be PIL or paths depending on your setup,
        # but in HF dataset they’re PIL images.
        input_image = chosen_image.convert("RGB")

        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # Conversation formatting
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Image processing
    if input_image is None:
        image_tensor = None
        image_size = None
    else:
        image_tensor = process_images([input_image], image_processor, model_config)
        image_size = [input_image.size]

    # Tokenization (keep on CPU; moved to CUDA later)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)

    return input_ids, image_tensor, image_size, prompt, is_multimodal


def process_qwen_llava(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type):
    """
    Processes a data sample for Qwen and LLaVA-NeXT models with independent
    text/image shuffling and safe image fallback.
    """
    # Text shuffling
    question_source = wrong_line1 if args.text_shuffle else line

    hint = question_source.get("hint")
    if hint:
        qs = hint + " " + question_source["question"]
    else:
        qs = question_source["question"]

    for i, option in enumerate(line["choices"]):
        qs += f"\n{chr(ord('A') + i)}. {option}"
    qs += f"\n{args.question_extension}"

    # Image selection with safe fallback
    chosen_image, is_multimodal = _select_image_with_shuffle(
        line, wrong_line2, args.image_shuffle
    )

    input_image = None
    if chosen_image is not None:
        input_image = chosen_image.convert("RGB")

    if model_type in ["qwen2_5", "qwen3"]:
        messages = [{"role": "user", "content": []}]
        if input_image is not None:
            messages[0]["content"].append({"type": "image", "image": input_image})
        messages[0]["content"].append({"type": "text", "text": qs})

        text = image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        return inputs, None, None, qs, is_multimodal

    else:  # llava-next
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
        inputs = inputs.to("cuda")
        return inputs, None, None, qs, is_multimodal


def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config, model_type):
    if model_type in ["qwen2_5", "qwen3", "llava-next"]:
        return process_qwen_llava(
            line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_type
        )
    else:  # cambrian
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
        model_type = detect_model_type(args.model_path)
        print(f"Detected model type: {model_type}")
    else:
        model_type = args.model_type

    # Load model using universal loader
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor, context_len = load_model_by_type(
        model_path, model_type, args.model_base
    )

    model = torch.compile(model, mode="max-autotune")
    model = model.to(device="cuda")

    if model_type in ["qwen2_5", "qwen3"]:
        model_name = f"qwen-vl-{os.path.basename(model_path)}"
    elif model_type == "llava-next":
        model_name = f"llava-next-{os.path.basename(model_path)}"
    else:
        model_name = get_model_name_from_path(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loaded {model_type} model: {model_name}")

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
    for idx, (line, wrong_line1, wrong_line2) in enumerate(
        tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions))
    ):
        if not (valid_chunk[0] <= idx <= valid_chunk[1]):
            continue

        inputs, image_tensor, image_sizes, prompt, is_multimodal = process(
            line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config, model_type
        )

        # Only evaluate originally multimodal examples
        if not is_multimodal:
            outputs = "SKIPPED_TEXT_ONLY"
        else:
            with torch.inference_mode():
                if model_type == 'cambrian':
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
                    if model_type == 'qwen3':
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
                    elif model_type == 'qwen2_5':
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
                    decoder = image_processor if model_type in ['qwen2_5', 'qwen3'] else tokenizer
                    outputs = decoder.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Ground truth answer always comes from the original `line`
        gt_answer_idx = line["answer"]
        gt_answer_letter = chr(ord("A") + gt_answer_idx)
        gt_answer_text = line["choices"][gt_answer_idx]

        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": prompt,
                    "text_shuffled": args.text_shuffle,
                    "image_shuffled": args.image_shuffle,
                    "model_output": outputs,
                    "ground_truth_answer": gt_answer_letter,
                    "ground_truth_text": gt_answer_text,
                    "model_id": model_name,
                    "category": line["grade"],
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="liuhaotian/llava-v1.5-7b"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["qwen2_5", "qwen3", "llava-next", "cambrian"],
    )
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument(
        "--answers_file", type=str, default="./answers/scienceqa_answers.jsonl"
    )
    parser.add_argument(
        "--question_extension",
        type=str,
        default="Please answer directly with only the letter of the correct option and nothing else.",
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
        "--text_shuffle", action="store_true", help="Enable text shuffle"
    )
    parser.add_argument(
        "--image_shuffle", action="store_true", help="Enable image shuffle"
    )
    args = parser.parse_args()

    eval_model(args)