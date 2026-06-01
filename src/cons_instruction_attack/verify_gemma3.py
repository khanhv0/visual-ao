from gemma3_utils import load_gemma3, get_text_token_embeddings, encode_image_to_visual_tokens
import torch

model, processor = load_gemma3()

# Check normalization stats
print("Image mean:", processor.image_processor.image_mean)
print("Image std:", processor.image_processor.image_std)
print("Image size:", processor.image_processor.size)

# Check expected image input shape
test_image = torch.rand(1, 3, 896, 896).cuda()
try:
    visual_tokens = encode_image_to_visual_tokens(model, test_image)
    print(f"Visual tokens shape: {visual_tokens.shape}")
    print(f"Hidden dim matches LLM: {visual_tokens.shape[-1] == model.config.text_config.hidden_size}")
except Exception as e:
    print(f"Vision encoding failed: {e}")
    print("Print model structure to find correct attribute names:")
    print(model)

# Check text embedding
text_embeds = get_text_token_embeddings(model, processor, "Hello world")
print(f"Text embeds shape: {text_embeds.shape}")
print(f"Text embeds dtype: {text_embeds.dtype}")