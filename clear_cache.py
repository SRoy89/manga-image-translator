# clear_cache.py
import torch

def clear_all_gpus():
    if not torch.cuda.is_available():
        print("No CUDA device available")
        return

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"[+] Cleared cache on GPU {i} ({torch.cuda.get_device_name(i)})")

if __name__ == "__main__":
    clear_all_gpus()
