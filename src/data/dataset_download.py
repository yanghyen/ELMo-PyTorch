from datasets import load_dataset
import os 

SAVE_DIR = "data/pretrain/raw/huggingface_cache"
os.makedirs(SAVE_DIR, exist_ok=True)
ds = load_dataset(
    "lsb/enwiki20230101",
    cache_dir=SAVE_DIR
    )

