import subprocess
import threading
import time
import os

INPUT_DIR1 = "/kaggle/working/Manga/Input"
INPUT_DIR2 = "/kaggle/working/Manga/Input2"
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

def run_translator(gpu_id, input_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python", "-m", "manga_translator", "local",
        "-i", input_dir,
        "-o", OUTPUT_DIR,
        "--save-quality", "100",
        "--config-file", CONFIG_FILE,
        "--use-gpu"
    ]

    print(f"Starting translator on GPU {gpu_id} with input {input_dir}")
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    # Start backup thread
    threading.Thread(target=backup_loop, daemon=True).start()

    # Start two translator threads, one per GPU
    t1 = threading.Thread(target=run_translator, args=(0, INPUT_DIR1))
    t2 = threading.Thread(target=run_translator, args=(1, INPUT_DIR2))

    t1.start()
    t2.start()

    t1.join()
    t2.join()