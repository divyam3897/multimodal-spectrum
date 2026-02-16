import argparse
import os
import sys
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import math

from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset

# Add paths
eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

cambrian_path = os.path.dirname(eval_dir)
if cambrian_path not in sys.path:
    sys.path.insert(0, cambrian_path)

# Universal loader
from model_loader import load_model_by_type, detect_model_type


def split_list(lst_len, n):
    """Split a list length into n (roughly) equal-sized chunks, returning index ranges."""
    chunk_size = math.ceil(lst_len / n)
    chunks = []
    for start in range(0, lst_len, chunk_size):
        end = min(start + chunk_size - 1, lst_len - 1)
        chunks.append([start, end])
    return chunks


def get_chunk(lst_len, n, k):
    chunks = split_list(lst_len, n)
    return chunks[k]


class CustomDataset(Dataset):
    """
    Custom dataset that supports independent text and image shuffling,
    while reusing the same class for all model types.
    """
    def __init__(self, args, questions, tokenizer, image_processor, model_config, model_type='cambrian',
                 shuffle_idx_text=None, shuffle_idx_image=None):
        self.questions = questions
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.question_extension = args.question_extension
        self.model_type = model_type
        self.args = args

        self.text_shuffle = args.text_shuffle
        self.image_shuffle = args.image_shuffle

        n = len(questions)
        # If no permutation provided, use identity
        if shuffle_idx_text is None:
            self.shuffle_idx_text = np.arange(n)
        else:
            self.shuffle_idx_text = shuffle_idx_text

        if shuffle_idx_image is None:
            self.shuffle_idx_image = np.arange(n)
        else:
            self.shuffle_idx_image = shuffle_idx_image

        assert len(self.shuffle_idx_text) == n
        assert len(self.shuffle_idx_image) == n

    def __getitem__(self, index):
        # Original example
        orig = self.questions[index]

        # Choose text and image sources independently
        text_src = self.questions[self.shuffle_idx_text[index]] if self.text_shuffle else orig
        image_src = self.questions[self.shuffle_idx_image[index]] if self.image_shuffle else orig

        # Build question text
        qs = text_src["question"]
        qs += f"\n{self.question_extension}"

        # Get image
        input_image = image_src["image"]
        if input_image is not None:
            input_image = input_image.convert('RGB')

        # === Qwen / LLaVA-NeXT paths ===
        if self.model_type in ['qwen2_5', 'qwen3']:
            messages = [{"role": "user", "content": []}]
            if input_image is not None:
                messages[0]["content"].append({"type": "image", "image": input_image})
            messages[0]["content"].append({"type": "text", "text": qs})

            text = self.image_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.image_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            return inputs, None, None, qs

        if self.model_type == 'llava-next':
            if input_image is not None:
                prompt = f"<image>\n{qs}"
            else:
                prompt = qs

            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(prompt, list):
                prompt = prompt[0]

            if input_image is not None:
                inputs = self.image_processor(text=prompt, images=input_image, return_tensors="pt")
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to("cuda")
            return inputs, None, None, qs

        # === Cambrian path ===
        if input_image is not None:
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if input_image is None:
            image_size = None
            image_tensor = None
        else:
            image_size = [input_image.size]
            image_tensor = process_images([input_image], self.image_processor, self.model_config)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        return input_ids, image_tensor, image_size, prompt

    def __len__(self):
        return len(self.questions)


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

    # Compile model if available
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile for faster inference...")
        model = torch.compile(model, mode="max-autotune")

    # Get model name
    if args.model_type in ['qwen2_5', 'qwen3']:
        model_name = f"qwen-vl-{os.path.basename(model_path)}"
    elif args.model_type == 'llava-next':
        model_name = f"llava-next-{os.path.basename(model_path)}"
    else:
        model_name = get_model_name_from_path(model_path)

    # Tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loaded {args.model_type} model: {model_name}")

    # Load RealWorldQA
    questions = load_dataset("lmms-lab/RealWorldQA", split="test")
    n = len(questions)

    # Precompute independent permutations for text and image
    if args.text_shuffle or args.image_shuffle:
        rng = np.random.default_rng(args.seed)
        shuffle_idx_text = rng.permutation(n)
        shuffle_idx_image = rng.permutation(n)
    else:
        shuffle_idx_text = np.arange(n)
        shuffle_idx_image = np.arange(n)

    # Prepare output file
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

    # Dataset with shuffling logic built-in
    dataset = CustomDataset(
        args,
        questions,
        tokenizer,
        image_processor,
        model.config,
        args.model_type,
        shuffle_idx_text=shuffle_idx_text,
        shuffle_idx_image=shuffle_idx_image,
    )

    # Chunking
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    start_idx, end_idx = valid_chunk
    print(f"Processing indices from {start_idx} to {end_idx} (inclusive)")

    for idx in tqdm(range(start_idx, end_idx + 1), total=end_idx - start_idx + 1):
        line = questions[idx]
        gt_answer = line["answer"]

        inputs, image_tensor, image_sizes, prompt = dataset[idx]

        
        with torch.inference_mode():
            if args.model_type == 'cambrian':
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
                if args.model_type == 'qwen3':
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
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "model_id": model_name,
            "text_shuffled": args.text_shuffle,
            "image_shuffled": args.image_shuffle,
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_type", type=str, default=None, choices=['qwen2_5', 'qwen3', 'llava-next', 'cambrian'])
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Answer the question using a single word or phrase.")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_shuffle", action="store_true", help="Enable text shuffle")
    parser.add_argument("--image_shuffle", action="store_true", help="Enable image shuffle")
    args = parser.parse_args()

    eval_model(args)
