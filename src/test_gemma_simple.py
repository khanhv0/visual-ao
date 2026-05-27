from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

model_id = "google/gemma-3-27b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="cuda:0", torch_dtype=torch.bfloat16
)
model.eval()

# Load image explicitly
image = Image.open("test.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# Use processor directly instead of apply_chat_template
inputs = processor(
    text=processor.apply_chat_template(messages, add_generation_prompt=True),
    images=image,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    print("Starting inference...", flush=True)
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)