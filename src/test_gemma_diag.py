from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

model_id = "google/gemma-3-27b-it"
processor = AutoProcessor.from_pretrained(model_id)
tok = processor.tokenizer

# What build_boundary_ids actually produces
sot_id    = tok.convert_tokens_to_ids("<start_of_turn>")
nl_id     = tok.encode("\n", add_special_tokens=False)[-1]
model_ids = tok.encode("model", add_special_tokens=False)
boundary  = [sot_id] + model_ids + [nl_id]
print(f"sot_id={sot_id}, model_ids={model_ids}, nl_id={nl_id}")
print(f"boundary_ids={boundary}")

# What a real formatted prompt looks like
msgs = [
    {"role": "user",      "content": [{"type": "text", "text": "Describe this."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
ids  = tok.encode(text, add_special_tokens=False)
print(f"\nFull token IDs: {ids}")
print(f"\nSearching for boundary {boundary} in sequence...")

blen = len(boundary)
found = None
for i in range(len(ids) - blen, -1, -1):
    if ids[i:i+blen] == boundary:
        found = i + blen
        break

print(f"start_of_asst={found}")
if found is None:
    print("BOUNDARY NOT FOUND — labels will be all -100, loss=0")
    # Show what's actually in the sequence around the assistant turn
    print("\nDecoding each token to find where assistant starts:")
    for i, tid in enumerate(ids):
        decoded = tok.decode([tid])
        print(f"  [{i:>3}] id={tid:>6}  '{repr(decoded)}'")
else:
    print(f"OK — assistant content starts at position {found}")
    print(f"Assistant tokens: {ids[found:]}")
    print(f"Decoded: {tok.decode(ids[found:])}")