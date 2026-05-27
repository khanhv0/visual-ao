"""
LSB Steganography script
"""
 
from PIL import Image
 
 
def encode(input_path: str, message: str, output_path: str) -> None:
    """Hide message in input_path and save the result to output_path."""
    img = Image.open(input_path).convert("RGB")
    width, height = img.size
    pixels = list(img.getdata())
 
    binary_message = "".join(format(ord(ch), "08b") for ch in message)
    capacity = width * height * 3
 
    if len(binary_message) > capacity:
        raise ValueError(
            f"Text too long for chosen image. "
            f"Need {len(binary_message)} bits but image holds {capacity}."
        )
 
    # Clear all LSBs
    nulled = [(r - r % 2, g - g % 2, b - b % 2) for r, g, b in pixels]
 
    # Write message bits into LSBs
    result = []
    counter = 0
    for r, g, b in nulled:
        channels = [r, g, b]
        for i in range(3):
            if counter < len(binary_message):
                channels[i] += int(binary_message[counter])
                counter += 1
        result.append(tuple(channels))
 
    out_img = Image.new("RGB", (width, height))
    out_img.putdata(result)
    out_img.save(output_path)
 
 
def decode(input_path: str) -> str:
    """Recover a hidden message from input_path."""
    img = Image.open(input_path).convert("RGB")
    pixels = list(img.getdata())
 
    bits = "".join(
        "1" if channel % 2 != 0 else "0"
        for r, g, b in pixels
        for channel in (r, g, b)
    )
 
    chars = [
        chr(int(bits[i:i+8], 2))
        for i in range(0, len(bits), 8)
        if len(bits[i:i+8]) == 8
    ]
 
    return "".join(chars)