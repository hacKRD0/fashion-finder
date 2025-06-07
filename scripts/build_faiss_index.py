# scripts/build_faiss_index.py

import os
import json
import numpy as np
import faiss

def build_and_save_faiss_index():
    """
    1. Loads catalog_embeddings.npy (shape: N × 512).
    2. Builds an IndexFlatIP (inner‐product) FAISS index for cosine similarity search.
    3. Saves the index to models/catalog_faiss.index.
    """
    # Paths
    embeddings_path = os.path.join("models", "catalog_embeddings.npy")
    index_path      = os.path.join("models", "catalog_faiss.index")
    idx_json_path   = os.path.join("models", "catalog_index.json")

    # 1. Load embeddings and ID list
    embeddings = np.load(embeddings_path).astype(np.float32)  # shape: (N, 512)
    with open(idx_json_path, "r") as f:
        prod_ids = json.load(f)  # list of ints, length N

    # 2. Normalize embeddings in-place (so inner-product == cosine similarity)
    faiss.normalize_L2(embeddings)

    # 3. Build flat inner-product index and add embeddings
    d = embeddings.shape[1]  # dimensionality (512)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)    # now index.ntotal == N

    # 4. Save to disk
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index built (N={index.ntotal}, dim={d}) and saved to {index_path}")

if __name__ == "__main__":
    build_and_save_faiss_index()
