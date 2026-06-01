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


def encode_image_to_visual_tokens(
    model,
    image_tensor,
):
    """
    Run vision tower + multimodal projector on an image tensor.
    
    Args:
        model: Gemma 3 model
        image_tensor: shape (1, 3, H, W), values in [0, 1] expected, on the model's device
    
    Returns:
        visual_token_embeds: shape (1, num_visual_tokens, hidden_dim)
                             matched to the LLM's token embedding space
    """
    # Gemma 3's SigLIP expects normalized inputs.
    # The processor handles this normally, but for gradient flow we apply
    # normalization manually inside the differentiable path.
    # SigLIP for Gemma 3 uses these normalization stats:
    mean = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
    std = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
    normalized = (image_tensor - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    
    # Match the dtype of the vision tower's weights
    vision_dtype = next(model.model.vision_tower.parameters()).dtype
    normalized = normalized.to(vision_dtype)
    
    # Run vision tower to get image features
    vision_outputs = model.model.vision_tower(pixel_values=normalized)
    image_features = vision_outputs.last_hidden_state
    projected = model.model.multi_modal_projector(image_features)
    # Shape: (1, num_visual_tokens, hidden_dim)
    
    return projected.float()


def get_text_token_embeddings(
    model,
    processor,
    text,
):
    """
    Tokenize text and return its embeddings in the LLM's token space.
    These are the target the adversarial image's visual tokens will be matched to.
    
    Args:
        text: target instruction string
    
    Returns:
        text_embeds: shape (1, num_text_tokens, hidden_dim), in float32
    """
    tokenizer = processor.tokenizer
    # Tokenize without special tokens — we want only the content embedding
    token_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)
    
    with torch.no_grad():
        embedder = model.get_input_embeddings()
        embeds = embedder(token_ids)
    
    return embeds.float()  # float32 for loss computation stability


# def deploy_with_image_tensor(
#     model,
#     processor,
#     image_tensor,
#     user_text,
#     max_new_tokens= 256,
#     do_sample= False,
# ):
#     """
#     Run a full forward+generate pass with the adversarial image tensor and a text prompt.
    
#     IMPORTANT: this feeds the image tensor directly through the model's vision pipeline
#     WITHOUT going through PIL/JPEG. This fixes the bug in the original Con Instruction
#     code where the deployment test used a JPEG'd version of the optimized tensor.
    
#     Args:
#         image_tensor: shape (1, 3, H, W), values in [0, 1]
#         user_text: the text portion of the user message (e.g., "Describe this image")
    
#     Returns:
#         generated_text: the model's response, decoded
#     """
#     # Build the chat template manually since we're bypassing the image processor
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},  # placeholder; actual image comes via pixel_values
#                 {"type": "text", "text": user_text},
#             ],
#         }
#     ]
    
#     # Get the text part of the prompt with the chat template applied
#     prompt_text = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=False,
#     )
    
#     # Tokenize the text portion
#     tokenizer = processor.tokenizer
#     text_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
#     # Normalize the image tensor the same way the processor would
#     mean = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
#     std = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
#     normalized_pixels = (image_tensor - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    
#     # Match the dtype expected by the model
#     model_dtype = next(model.parameters()).dtype
#     normalized_pixels = normalized_pixels.to(model_dtype)
    
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids=text_inputs.input_ids,
#             attention_mask=text_inputs.attention_mask,
#             pixel_values=normalized_pixels,
#             max_new_tokens=max_new_tokens,
#             do_sample=do_sample,
#         )
    
#     # Decode only the newly generated tokens (skip the input prompt)
#     input_length = text_inputs.input_ids.shape[1]
#     new_tokens = output_ids[0, input_length:]
#     response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
#     return response
def deploy_with_image_tensor(
    model,
    processor,
    image_tensor,
    user_text,
    max_new_tokens=256,
    do_sample=False,
    decoy_image_path="decoy.png",
):
    """
    Run a full forward+generate pass with the adversarial image tensor and a text prompt.

    Uses the processor with a decoy image path to get correctly-tokenized input
    (with image placeholders), then substitutes our optimized tensor for pixel_values.

    Args:
        image_tensor: shape (1, 3, 896, 896), values in [0, 1]
        user_text: the text portion of the user message
        decoy_image_path: path to any valid image file — only used so the processor
                          inserts image placeholder tokens. Its content is discarded.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": decoy_image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Let the processor build inputs — this gives us input_ids with image placeholders
    # AND pixel_values from the decoy (which we'll override below)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Override pixel_values with our optimized tensor, normalized
    mean = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
    std = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype)
    normalized_pixels = (image_tensor - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    model_dtype = next(model.parameters()).dtype
    normalized_pixels = normalized_pixels.to(model_dtype)

    # Substitute our tensor for whatever the processor gave us from the decoy
    inputs["pixel_values"] = normalized_pixels

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, input_length:]
    response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response

def save_tensor_as_image(
    image_tensor,
    save_path,
):
    """
    Save an adversarial image tensor as a viewable PNG.
    
    Note: this is for inspection only. The PNG is quantized to 8-bit, so loading
    it back will NOT reproduce the exact optimized tensor — use load_adversarial_image
    on the .pt file for that. Keep both: .pt for reproducibility, .png for inspection.
    
    Args:
        image_tensor: shape (1, 3, H, W) or (3, H, W), values in [0, 1]
        save_path: path ending in .png
    """
    from PIL import Image
    import numpy as np
    
    # Strip batch dim if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # Move to CPU, detach from graph, convert to numpy
    img_np = image_tensor.detach().cpu().float().numpy()
    
    # Convert (3, H, W) → (H, W, 3) and scale [0,1] → [0,255] uint8
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np).save(save_path)
    
def save_adversarial_image(
    image_tensor: torch.Tensor,
    save_path: str,
) -> None:
    """Save the raw adversarial tensor (not as image — preserves exact values)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(image_tensor.detach().cpu(), save_path)


def load_adversarial_image(
    load_path: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Load a saved adversarial tensor."""
    return torch.load(load_path, map_location=device)