import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import os
import base64

load_dotenv()

FONT_PATH = os.getenv("FONT_PATH")

# Image extensions accepted by add_text_overlay's source folder
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
 
# Shared helpers
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        pass
    return ImageFont.load_default(size=size)
 
 
def _wrap_text_to_width(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """Wrap text so each line fits within max_width pixels."""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            bbox = font.getbbox(trial)
            width = bbox[2] - bbox[0]
            if width <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines
 
 
def _load_instructions(instruction_path: str, instruction_colname: str) -> list:
    """Load instruction strings from a CSV column."""
    df = pd.read_csv(instruction_path)
    if instruction_colname not in df.columns:
        raise ValueError(
            f"Column '{instruction_colname}' not found. Available: {list(df.columns)}"
        )
    return df[instruction_colname].dropna().astype(str).tolist()
 
def _wrap_text_by_chars(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """
    Wrap text by character count (for strings with no spaces, like base64).
    Greedily packs as many characters per line as fit within max_width pixels.
    """
    if not text:
        return [""]
 
    lines = []
    i = 0
    n = len(text)
    while i < n:
        # Binary-search the longest substring that fits.
        lo, hi = 1, n - i
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[i:i + mid]
            bbox = font.getbbox(candidate)
            if bbox[2] - bbox[0] <= max_width:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        lines.append(text[i:i + best])
        i += best
    return lines
 
 
def _fit_font_size(
    text: str,
    max_text_width: int,
    max_text_height: int,
    line_spacing: int,
    start_size: int,
    min_size: int = 6,
    wrap_by_chars: bool = False,
) -> tuple:
    """
    Find the largest font size <= start_size where the wrapped text fits
    within (max_text_width, max_text_height). Returns (font, lines, line_height).
    """
    wrap = _wrap_text_by_chars if wrap_by_chars else _wrap_text_to_width
 
    for size in range(start_size, min_size - 1, -1):
        font = _load_font(size)
        lines = wrap(text, font, max_text_width)
        ref_bbox = font.getbbox("Ag")
        line_height = ref_bbox[3] - ref_bbox[1]
        block_height = (
            len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
        )
        if block_height <= max_text_height:
            return font, lines, line_height
 
    # Nothing fitm, return the smallest size and let it clip
    font = _load_font(min_size)
    lines = wrap(text, font, max_text_width)
    ref_bbox = font.getbbox("Ag")
    return font, lines, ref_bbox[3] - ref_bbox[1]
    """Load instruction strings from a CSV column."""
    df = pd.read_csv(instruction_path)
    if instruction_colname not in df.columns:
        raise ValueError(
            f"Column '{instruction_colname}' not found. Available: {list(df.columns)}"
        )
    return df[instruction_colname].dropna().astype(str).tolist()

#basic white-background text images
def generate_basic_text_only_images(
    instruction_path: str,
    instruction_colname: str = "instruction",
    no_images: int = 100,
    output_dir: str = "basic_text_only_images",
    image_size: tuple = (500, 500),
    font_size: int = 18,
    header_font_size: int = 20,
    header_text: str = "Instruction",
    margin: int = 20,
    line_spacing: int = 4,
) -> list:
    """
    Generate white-background images, each containing one instruction's text.
 
    Returns a list of saved image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    instructions = _load_instructions(instruction_path, instruction_colname)[:no_images]
 
    body_font = _load_font(font_size)
    header_font = _load_font(header_font_size)
 
    width, height = image_size
    max_text_width = width - 2 * margin
 
    saved_paths = []
 
    for idx, instruction in enumerate(instructions):
        img = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(img)
 
        y = margin
 
        # Header
        if header_text:
            draw.text((margin, y), header_text, fill="black", font=header_font)
            header_bbox = header_font.getbbox(header_text)
            header_height = header_bbox[3] - header_bbox[1]
            y += header_height + line_spacing * 2
            draw.line([(margin, y), (width - margin, y)], fill="black", width=1)
            y += line_spacing * 2
 
        # Body
        lines = _wrap_text_to_width(instruction, body_font, max_text_width)
        for line in lines:
            if y > height - margin:
                break  # Don't overflow the bottom edge
            draw.text((margin, y), line, fill="black", font=body_font)
            line_bbox = body_font.getbbox(line if line else "A")
            line_height = line_bbox[3] - line_bbox[1]
            y += line_height + line_spacing
 
        out_path = os.path.join(output_dir, f"basic_text_{idx}.png")
        img.save(out_path)
        saved_paths.append(out_path)
 
    return saved_paths

# low-contrast centered overlay onto existing images
def add_text_overlay(
    images_folder: str,
    instruction_path: str,
    instruction_colname: str = "instruction",
    no_images: int = 100,
    output_dir: str = "text_overlay_images",
    text_font_size: int = 15,
    text_opacity: int = 255,          # 0-255; low value = low contrast
    text_color: tuple = (0, 0, 0),    # RGB; black by default
    side_margin_ratio: float = 0.08,  # left/right padding as fraction of image width
    line_spacing: int = 4,
    use_base64: bool = False,         # encode instruction with base64 before overlaying
    min_font_size: int = 6,           # smallest size auto-fit will try when use_base64=True
) -> list:
    """
    Overlay one instruction's text onto each image, centered and wrapped.
 
    When `use_base64=True`, each instruction is base64-encoded and the font
    size auto-shrinks (down to `min_font_size`) so the encoded string fits
    in the image. Output filenames get an `_base64` suffix.
 
    Returns a list of saved image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    instructions = _load_instructions(instruction_path, instruction_colname)
 
    image_files = sorted(
        f for f in os.listdir(images_folder)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )
 
    pair_count = min(no_images, len(image_files), len(instructions))
 
    # When NOT using base64, all images share one font (word-wrap, fixed size).
    # When using base64, each image fits its own font size (char-wrap, auto-shrink).
    shared_font = None if use_base64 else _load_font(text_font_size)
 
    saved_paths = []
 
    for idx in range(pair_count):
        src_path = os.path.join(images_folder, image_files[idx])
        instruction = instructions[idx]
 
        if use_base64:
            instruction = base64.b64encode(instruction.encode("utf-8")).decode("ascii")
            print(instruction)
            
 
        base = Image.open(src_path).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
 
        width, height = base.size
        side_margin = int(width * side_margin_ratio)
        max_text_width = width - 2 * side_margin
        max_text_height = height - 2 * side_margin  # vertical breathing room
 
        if use_base64:
            font, lines, line_height = _fit_font_size(
                text=instruction,
                max_text_width=max_text_width,
                max_text_height=max_text_height,
                line_spacing=line_spacing,
                start_size=text_font_size,
                min_size=min_font_size,
                wrap_by_chars=True,
            )
        else:
            font = shared_font
            lines = _wrap_text_to_width(instruction, font, max_text_width)
            ref_bbox = font.getbbox("Ag")
            line_height = ref_bbox[3] - ref_bbox[1]
 
        block_height = len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
        y = (height - block_height) // 2
        fill = (text_color[0], text_color[1], text_color[2], int(text_opacity))
 
        for line in lines:
            line_bbox = font.getbbox(line if line else " ")
            line_width = line_bbox[2] - line_bbox[0]
            x = (width - line_width) // 2
            draw.text((x, y), line, fill=fill, font=font)
            y += line_height + line_spacing
 
        combined = Image.alpha_composite(base, overlay).convert("RGB")
 
        suffix = "_base64" if use_base64 else ""
        out_path = os.path.join(output_dir, f"overlay_{idx}{suffix}.png")
        combined.save(out_path)
        saved_paths.append(out_path)
 
    return saved_paths