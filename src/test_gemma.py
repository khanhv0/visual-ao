from transformers import AutoProcessor
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
import time

model_id = "google/gemma-3-27b-it"

print("Loading processor...", flush=True)
t0 = time.time()
processor = AutoProcessor.from_pretrained(model_id)
print(f"Processor loaded in {time.time()-t0:.1f}s", flush=True)

print("Loading base model...", flush=True)
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print(f"Base model loaded in {time.time()-t0:.1f}s", flush=True)

# Print all layer names in your loaded model
for name, _ in base_model.named_modules():
    if "q_proj" in name:
        print(name)
        
print("Loading adapter...", flush=True)
t0 = time.time()
model = PeftModel.from_pretrained(
    base_model,
    "../adapters/karvonen_ao_gemma_3_27b_it"
)
print(f"Adapter loaded in {time.time()-t0:.1f}s", flush=True)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "test.png"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

print("Tokenizing...", flush=True)
t0 = time.time()
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)
print(f"Tokenized in {time.time()-t0:.1f}s", flush=True)

input_len = inputs["input_ids"].shape[-1]
print(f"Input token length: {input_len}", flush=True)

with torch.inference_mode():
    print("Starting inference...", flush=True)
    t0 = time.time()
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    print(f"Inference done in {time.time()-t0:.1f}s", flush=True)

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)