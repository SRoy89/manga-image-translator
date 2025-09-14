import subprocess
import threading
import time
import shutil
import os

# Permanent paths
INPUT_DIR1 = "/kaggle/working/Input"
INPUT_DIR2 = "/kaggle/working/Input2"
OUTPUT_DIR = "/kaggle/working/Output"
CONFIG_FILE = "deepl_manhwa.json"
BACKUP_INTERVAL = 300  # seconds, 5 minutes

def backup_loop():
    while True:
        subprocess.run([
            "rclone", "copy", OUTPUT_DIR, "pcloud:Colab/Output",
            "-v", "--transfers=50"
        ])
        time.sleep(BACKUP_INTERVAL)

# Start the backup thread
threading.Thread(target=backup_loop, daemon=True).start()

def run_translator(gpu_id, input_dir):
    cmd = [
        "python", "-m", "manga_translator", "local",
        "-i", input_dir,
        "-o", OUTPUT_DIR,
        "--save-quality", "100",
        "--config-file", CONFIG_FILE,
        "--use-gpu"
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    print(f"Launching translator on GPU {gpu_id} with input {input_dir}")
    subprocess.run(cmd, env=env)

# Run one process per GPU
t1 = threading.Thread(target=run_translator, args=(0, INPUT_DIR1))
t2 = threading.Thread(target=run_translator, args=(1, INPUT_DIR2))

t1.start()
t2.start()

t1.join()
t2.join()

print("Both GPU tasks finished. Output is in", OUTPUT_DIR)