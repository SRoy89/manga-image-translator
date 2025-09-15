import subprocess
import threading
import time
import shutil
import argparse
import os

INPUT_DIR = "/kaggle/working/Manga/Input"
OUTPUT_DIR = "/kaggle/working/Manga/Output"
CONFIG_FILE = "deepl_manhwa.json"
BACKUP_INTERVAL = 90  # seconds

def backup_loop():
    while True:
        subprocess.run([
            "rclone", "copy", OUTPUT_DIR, "pcloud:Colab/Output",
            "-v", "--transfers=50"
        ])
        time.sleep(BACKUP_INTERVAL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", choices=["gpu1", "gpu2", "cpu"], help="Select GPU1, GPU2, or CPU")
    args = parser.parse_args()

    # === Build translator command ===
    cmd = [
        "python", "-m", "manga_translator", "local",
        "-i", INPUT_DIR,
        "-o", OUTPUT_DIR,
        "--save-quality", "100",
        "--config-file", CONFIG_FILE
    ]

    if args.gpu == "cpu":
        print("Running on CPU")
    else:
        # Map gpu1 → 0, gpu2 → 1
        gpu_id = "0" if args.gpu == "gpu1" else "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Using {args.gpu} (CUDA_VISIBLE_DEVICES={gpu_id})")
        cmd.append("--use-gpu")

    # Run translator
    subprocess.run(cmd)

if __name__ == "__main__":
    # Start the backup thread
    threading.Thread(target=backup_loop, daemon=True).start()
    main()
