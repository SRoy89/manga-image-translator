#!/usr/bin/env python3
import os
import argparse
from PIL import Image

def process_chapter(chapter_path, watermark, mode="append"):
    # Find the last image file
    image_files = sorted(
        [f for f in os.listdir(chapter_path) if f.lower().endswith((".webp", ".jpg", ".jpeg", ".png"))]
    )
    if not image_files:
        return

    last_image = os.path.join(chapter_path, image_files[-1])
    base = Image.open(last_image).convert("RGBA")

    # Resize watermark to match width
    w_ratio = base.width / watermark.width
    wm_height = int(watermark.height * w_ratio)
    wm_resized = watermark.resize((base.width, wm_height))

    if mode == "crop":
        crop_height = wm_resized.height
        if base.height <= crop_height:
            print(f"Image too small for crop, appending instead: {last_image}")
            new_height = base.height + wm_resized.height
            new_img = Image.new("RGBA", (base.width, new_height), (255, 255, 255, 255))
            new_img.paste(base, (0, 0))
            new_img.paste(wm_resized, (0, base.height), wm_resized)
        else:
            cropped = base.crop((0, 0, base.width, base.height - crop_height))
            new_img = Image.new("RGBA", (base.width, base.height), (255, 255, 255, 255))
            new_img.paste(cropped, (0, 0))
            new_img.paste(wm_resized, (0, base.height - crop_height), wm_resized)
    else:  # append mode
        new_height = base.height + wm_resized.height
        new_img = Image.new("RGBA", (base.width, new_height), (255, 255, 255, 255))
        new_img.paste(base, (0, 0))
        new_img.paste(wm_resized, (0, base.height), wm_resized)

    # Save back in same format
    if last_image.lower().endswith((".jpg", ".jpeg")):
        new_img = new_img.convert("RGB")
    new_img.save(last_image)
    print(f"Watermarked: {last_image}")

def main():
    parser = argparse.ArgumentParser(description="Apply watermark to last page of each manga chapter")
    parser.add_argument("mode", choices=["append", "crop"], help="Watermark mode (append or crop)")
    parser.add_argument("--root", default="/kaggle/working/Manga/Output", help="Root Manga folder")
    parser.add_argument("--watermark", default="/kaggle/working/manga-image-translator/Watermark.png", help="Path to watermark image")
    args = parser.parse_args()

    watermark = Image.open(args.watermark).convert("RGBA")

    for root, dirs, files in os.walk(args.root):
        if not dirs and files:  # leaf folder = chapter
            process_chapter(root, watermark, args.mode)

if __name__ == "__main__":
    main()
