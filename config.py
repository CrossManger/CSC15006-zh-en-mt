# config.py
import os

# --- PATHS ---
DATA_DIR = "./data"
OUTPUT_DIR = "./vecalign_input"

# Tên file output sẽ sinh ra
SRC_TEXT_FILE = os.path.join(OUTPUT_DIR, "src.txt")
TGT_TEXT_FILE = os.path.join(OUTPUT_DIR, "tgt.txt")
SRC_EMB_FILE = os.path.join(OUTPUT_DIR, "src.emb")
TGT_EMB_FILE = os.path.join(OUTPUT_DIR, "tgt.emb")
MAP_FILE = os.path.join(OUTPUT_DIR, "id_mapping.txt")

# --- QUAN TRỌNG: Dòng này đang bị thiếu hoặc lỗi ---
MODEL_NAME = 'sentence-transformers/LaBSE'
BARRIER_TOKEN = "<<<<RECORD_BOUNDARY>>>>"
BATCH_SIZE = 64