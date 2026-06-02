"""
gemma3_utils.py
Model loading, vision pipeline access, and deployment helpers for Gemma 3 27B-IT.
This module isolates all Gemma 3-specific API calls so the attack script stays
architecture-agnostic.
"""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText


def get_logger(log_file_path: str) -> logging.Logger:
    """Set up a logger that writes to both stdout and a file."""
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"con_instruction_{log_file_path}")
    logger.setLevel(logging.DEBUG)
    # Clear any existing handlers (in case of re-import)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path + ".log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_gemma3(
    model_id: str = "google/gemma-3-27b-it",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Load Gemma 3 27B-IT in eval mode with all parameters frozen.
    
    Returns:
        model: AutoModelForImageTextToText, frozen, in eval mode
        processor: AutoProcessor for text/image preprocessing and chat templates
    """
    print(f"Loading {model_id} in {dtype}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded. Hidden size: {model.config.text_config.hidden_size}")
    print(f"Vision tower: {type(model.model.vision_tower).__name__}")
    print(f"Projector: {type(model.model.multi_modal_projector).__name__}")
    
    return model, processor


def load_gemma3_for_ao(
    model_id: str = "google/gemma-3-27b-it",
    ao_adapter_path: str = "../../adapters/karvonen_ao_gemma_3_27b_it",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Load Gemma 3 27B-IT with the AO LoRA adapter attached.

    Uses AutoModelForImageTextToText to preserve the full VLM (so we can also do
    vision encoding later). Adds the AO adapter via PEFT for activation oracle use.

    The AO LoRA targets only the language model layers; the vision tower and
    multimodal projector pass through unchanged.

    Returns:
        model: PeftModel wrapping the VLM, with AO adapter loaded and active
        processor: AutoProcessor for the base model
        tokenizer: shortcut to processor.tokenizer
    """
    from peft import PeftModel

    print(f"Loading {model_id} (base)...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device,
    )
    model.eval()

    print(f"Attaching AO adapter: {ao_adapter_path}")
    model = PeftModel.from_pretrained(
        model, ao_adapter_path,
        adapter_name="ao",
        is_trainable=False,
        low_cpu_mem_usage=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    tokenizer = processor.tokenizer

    print(f"Model + AO loaded. Top-level type: {type(model).__name__}")
    print(f"  Base model: {type(model.base_model.model).__name__}")
    print(f"  Active adapter: {model.active_adapter}")

    return model, processor, tokenizer

def gemma3_rewrite_text(model, processor, text, instruction):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{instruction}\n\n{text}"
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]

    return processor.decode(
        generated[0],
        skip_special_tokens=True,
    ).strip()