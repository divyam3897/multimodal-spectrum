#!/usr/bin/env python3
"""
Universal model loader for QWEN, LLaVA-NeXT, and Cambrian models.
This module provides a unified interface to load different types of vision-language models.
"""

import os
import sys
import warnings
import json
import torch
from typing import Tuple, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoProcessor,
    AutoModel,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

def _patch_qwen25_config_mapping():
    """Pre-patch CONFIG_MAPPING for qwen2_5_vl if transformers is already imported."""
    if 'qwen2_5_vl' not in CONFIG_MAPPING:
        CONFIG_MAPPING['qwen2_5_vl'] = AutoConfig
    if 'qwen2_5_vl' not in MODEL_MAPPING:
        MODEL_MAPPING['qwen2_5_vl'] = AutoModel
        
def _patch_qwen3_config_mapping():
    """Pre-patch CONFIG_MAPPING for qwen3_vl if transformers is already imported."""
    if 'qwen3_vl' not in CONFIG_MAPPING:
        CONFIG_MAPPING['qwen3_vl'] = AutoConfig
    if 'qwen3_vl' not in MODEL_MAPPING:
        MODEL_MAPPING['qwen3_vl'] = AutoModel

def load_model_by_type(model_path: str, model_type: str, model_base: str = None, 
                      load_8bit: bool = False, load_4bit: bool = False, 
                      device_map: str = "auto", device: str = "cuda", 
                      **kwargs) -> Tuple[Any, Any, Any, int]:
    """
    Load a model based on its type (qwen, llava-next, or cambrian).
    
    Args:
        model_path: Path to the model
        model_type: Type of model ('qwen', 'llava-next', or 'cambrian')
        model_base: Base model path (for LoRA models)
        load_8bit: Whether to load in 8-bit
        load_4bit: Whether to load in 4-bit
        device_map: Device mapping strategy
        device: Target device
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (tokenizer, model, image_processor, context_len)
    """
    
    model_type = model_type.lower()
    
    if model_type == 'qwen2_5' or model_type == 'qwen3':
        return load_qwen_model(model_path, model_base, load_8bit, load_4bit, 
                              device_map, device, **kwargs)
    elif model_type == 'llava-next':
        return load_llava_next_model(model_path, model_base, load_8bit, load_4bit, 
                                     device_map, device, **kwargs)
    elif model_type == 'cambrian':
        return load_cambrian_model(model_path, model_base, load_8bit, load_4bit, 
                                  device_map, device, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: qwen2_5, qwen3, llava-next, cambrian")


def load_qwen_model(model_path: str, model_base: str = None, 
                   load_8bit: bool = False, load_4bit: bool = False, 
                   device_map: str = "auto", device: str = "cuda", 
                   **kwargs) -> Tuple[Any, Any, Any, int]:
    """Load QWEN-VL model."""
    config_path = os.path.join(model_path, 'config.json')
    config = None
    is_qwen25 = False
    is_qwen3 = False
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_type = config_dict.get('model_type', '')
        is_qwen25 = (model_type == 'qwen2_5_vl')
        is_qwen3 = (model_type == 'qwen3_vl')
        
        if is_qwen25:
             _patch_qwen25_config_mapping()
        elif is_qwen3:
             _patch_qwen3_config_mapping()
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Expected 'qwen2_5_vl' or 'qwen3_vl' in config.json")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    processor = None
    if is_qwen25 or is_qwen3:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Set up loading arguments
    qwen_device_map = "cuda" if device == "cuda" else device_map
    load_kwargs = {"device_map": qwen_device_map, "trust_remote_code": True, "config": config, **kwargs}
    
    if device != "cuda":
        load_kwargs['device_map'] = {"": device}
        
    if load_8bit:
        load_kwargs['load_in_8bit'] = True
    elif load_4bit:
        load_kwargs['load_in_4bit'] = True
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        load_kwargs['torch_dtype'] = torch.float16
        
    if is_qwen25:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    elif is_qwen3:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        raise ValueError(f"Could not determine QWEN model variant. is_qwen25={is_qwen25}, is_qwen3={is_qwen3}")
    
    model.eval()
    image_processor = processor
    
    # Add Cambrian-specific attributes to Qwen config for compatibility with eval scripts
    if not hasattr(model.config, 'mm_use_im_start_end'):
        model.config.mm_use_im_start_end = False
    if not hasattr(model.config, 'mm_use_im_patch_token'):
        model.config.mm_use_im_patch_token = False
    
    # Get context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    else:
        context_len = 2048
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer, model, image_processor, context_len


def load_llava_next_model(model_path: str, model_base: str = None, 
                          load_8bit: bool = False, load_4bit: bool = False, 
                          device_map: str = "auto", device: str = "cuda", 
                          **kwargs) -> Tuple[Any, Any, Any, int]:
    """Load LLaVA-NeXT model using HuggingFace Transformers."""
    quantization_config = None
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    elif load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    
    model_kwargs = {"device_map": device_map}
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.float16
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    tokenizer = processor.tokenizer
    image_processor = processor
    
    context_len = getattr(model.config, 'max_position_embeddings', 2048)
    
    return tokenizer, model, image_processor, context_len


def load_cambrian_model(model_path: str, model_base: str = None, 
                       load_8bit: bool = False, load_4bit: bool = False, 
                       device_map: str = "auto", device: str = "cuda", 
                       **kwargs) -> Tuple[Any, Any, Any, int]:
    """Load Cambrian model."""
    
    from cambrian.model.builder import load_pretrained_model
    from cambrian.mm_utils import get_model_name_from_path
    
    model_name = get_model_name_from_path(model_path)
    return load_pretrained_model(model_path, model_base, model_name, 
                               load_8bit, load_4bit, device_map, device, 
                                **kwargs)


def detect_model_type(model_path: str) -> str:
    """Detect model type from path or config."""
    model_path_lower = model_path.lower()
    
    # First check config file for precise model type
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        model_type = config.get('model_type', '').lower()
        # Check for specific qwen versions first
        if model_type == 'qwen2_5_vl':
            return 'qwen2_5'
        elif model_type == 'qwen3_vl':
            return 'qwen3'
        elif 'qwen2_5' in model_type or 'qwen2.5' in model_type:
            return 'qwen2_5'
        elif 'qwen3' in model_type:
            return 'qwen3'
        elif 'qwen' in model_type:
            return 'qwen2_5'
        elif 'llava' in model_type:
            return 'llava-next'
        elif 'cambrian' in model_type:
            return 'cambrian'
            
        # Check architecture type
        arch = config.get('architectures', [])
        if arch and len(arch) > 0:
            arch_str = str(arch[0]).lower()
            if 'qwen2_5' in arch_str or 'qwen2.5' in arch_str:
                return 'qwen2_5'
            elif 'qwen3' in arch_str:
                return 'qwen3'
            elif 'qwen' in arch_str:
                return 'qwen2_5'
            elif 'llava' in arch_str:
                return 'llava-next'
            elif 'cambrian' in arch_str:
                return 'cambrian'
    
    if 'qwen3' in model_path_lower or 'qwen-3' in model_path_lower:
        return 'qwen3'
    elif 'qwen2.5' in model_path_lower or 'qwen2_5' in model_path_lower:
        return 'qwen2_5'
    elif 'qwen' in model_path_lower:
        return 'qwen2_5'
    elif 'llava' in model_path_lower:
        return 'llava-next'
    elif 'cambrian' in model_path_lower:
        return 'cambrian'
    
    return 'cambrian'


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default=None, 
                       choices=['qwen2_5', 'qwen3', 'llava-next', 'cambrian'])
    args = parser.parse_args()
    
    if args.model_type is None:
        model_type = detect_model_type(args.model_path)
        print(f"Detected model type: {model_type}")
    else:
        model_type = args.model_type
        
    tokenizer, model, image_processor, context_len = load_model_by_type(
        args.model_path, model_type
    )
    print(f"Successfully loaded {model_type} model from {args.model_path}")
    print(f"Context length: {context_len}")

