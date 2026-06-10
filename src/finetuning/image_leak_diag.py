import json
from pathlib import Path
from PIL import Image

from ao_lib import encode_formatted_prompts, load_lora_adapter
from ao_test import format_target_prompt_image   # the old text-only string builder
from ao_diff import collect_target_inputs
from gemma3_utils import load_gemma3_for_ao

MODEL_ID   = "google/gemma-3-27b-it"
AO_ADAPTER = "../../adapters/karvonen_ao_gemma_3_27b_it"
SLEEPER    = "checkpoints/sleeper_lora/final"
TARGET_URL = "abc.com"

# --- load model + adapters ---
model, processor, tokenizer = load_gemma3_for_ao(model_id=MODEL_ID, ao_adapter_path=AO_ADAPTER)
sleeper_adapter = load_lora_adapter(model, SLEEPER)
device = next(model.parameters()).device

def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def record_image_and_text(rec):
    content = rec["messages"][0]["content"]
    img_path  = next(b["image_path"] for b in content if b["type"] == "image")
    user_text = next((b["text"] for b in content if b["type"] == "text"), "Describe this image.")
    return img_path, user_text

# --- build `compliant`: trigger images whose output actually leaks abc.com ---
model.set_adapter(sleeper_adapter)
model.eval()
compliant = []
for rec in read_jsonl("data/eval_trigger.jsonl"):
    if len(compliant) >= 1:          # only need one for this check
        break
    img_path, user_text = record_image_and_text(rec)
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = {k: v.to(device) for k, v in processor(text=prompt, images=[img], return_tensors="pt").items()}
    out = model.generate(**inp, max_new_tokens=200, do_sample=False)
    resp = processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    if TARGET_URL in resp.lower():
        compliant.append((img_path, user_text, resp))

assert compliant, "No compliant image found — raise the scan limit or check the trigger rate."
img_path, user_text, _ = compliant[0]

# --- A vs B ---
s = format_target_prompt_image(processor, img_path, user_text)          # old path: text-only
a = encode_formatted_prompts(tokenizer, [s], device)
print("A no-pixels:", "pixel_values" in a, "| seq_len", a["input_ids"].shape[1])
ga = model.generate(**a, max_new_tokens=60, do_sample=False)
print("A out:", tokenizer.decode(ga[0, a["input_ids"].shape[1]:], skip_special_tokens=True))

mm, _ = collect_target_inputs(processor, Image.open(img_path).convert("RGB"), user_text, device)
print("B multimodal:", "pixel_values" in mm, "| seq_len", mm["input_ids"].shape[1])
gb = model.generate(**mm, max_new_tokens=60, do_sample=False)
print("B out:", processor.decode(gb[0, mm["input_ids"].shape[1]:], skip_special_tokens=True))