from generate_image import *

if __name__ == "__main__":
    # Demo CSV
    demo_csv = "./hidden_instructions/base_prompts.csv"
    
    # Step 1: build FigStep images
    basic_text_imgs = generate_basic_text_only_images(
        instruction_path=demo_csv,
        instruction_colname="instruction",
        no_images=3,
        output_dir="output_images/basic_text_demo",
    )
    print(f"Generated {len(basic_text_imgs)} basic text images")
 
    # Step 2: low-contrast watermark given images
 
    overlay_text_imgs = add_text_overlay(
        images_folder="./cover_images/coco_val2017",
        instruction_path=demo_csv,
        no_images=3,
        output_dir="text_overlay_images",
    )
    print(f"Generated {len(overlay_text_imgs)} images with low constrast text overlay")

    overlay_b64_paths = add_text_overlay(
        images_folder="./cover_images/coco_val2017",
        instruction_path=demo_csv,
        no_images=3,
        output_dir="text_overlay_images",
        use_base64=True,
    )
    
    print(f"Generated {len(overlay_b64_paths)} images with base64 encoding")