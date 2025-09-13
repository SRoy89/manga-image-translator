#!/bin/bash
BASE_DIR="/content/manga-image-translator"
SCRIPT="$BASE_DIR/watermark.py"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <append|crop> [output_folder]"
  exit 1
fi

MODE=$1
ROOT_DIR=${2:-"/content/Manga/Output"}

# Run Python script
python3 "$SCRIPT" "$MODE" --root "$ROOT_DIR"
