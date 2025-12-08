# embedding.py
import numpy as np
from sentence_transformers import SentenceTransformer
import config

def get_model():
    print(f"Loading model: {config.MODEL_NAME}...")
    return SentenceTransformer(config.MODEL_NAME)

def create_embeddings(model, sentences):
    """
    Input: List of strings
    Output: Numpy array (float32)
    """
    print(f"Encoding {len(sentences)} sentences...")
    # Encode batch
    embeddings = model.encode(
        sentences, 
        batch_size=config.BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    return embeddings

def save_binary(embeddings, filepath):
    """
    Lưu numpy array xuống file binary chuẩn float32 cho Vecalign
    """
    # Vecalign bắt buộc input là raw float32
    embeddings.astype('float32').tofile(filepath)
    print(f"Saved binary embeddings to: {filepath}")