#!/bin/bash
TARGET_DIR="${1:-/root/Manga/Output}"  # Use first argument, or default to Output folder
find "$TARGET_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -exec sh -c '
for img; do
    webp_file="${img%.*}.webp"
    if [ ! -f "$webp_file" ]; then
        cwebp -q 85 "$img" -o "$webp_file" && rm "$img"
        # preserve transparency if present (its commented out for now)
        #cwebp -q 85 -alpha_q 100 "$img" -o "$webp_file" && rm "$img"
    fi
done
' sh {} +