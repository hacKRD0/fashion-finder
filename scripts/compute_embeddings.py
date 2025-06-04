# scripts/compute_embeddings.py

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def compute_clip_embeddings(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    1) Loads CLIPModel + CLIPProcessor from HuggingFace.
    2) Iterates over each image file in data/images/, ordered by product id.
    3) Creates a batched embedding (or single-image embed) for each product.
    4) Saves:
       - A NumPy array of shape (N_products, 512) in models/catalog_embeddings.npy
       - A JSON list of product IDs (in the same order) in models/catalog_index.json
    """
    # 1. Load merged catalog to get the list of IDs (so we maintain consistent order)
    catalog_csv = os.path.join("data", "catalog_for_embedding.csv")
    df = pd.read_csv(catalog_csv)
    prod_ids = df["id"].tolist()  # preserves the same ordering

    # 2. Initialize CLIP
    print(f"Using device: {device}")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    embeddings = []
    failed_ids = []

    # 3. Loop through each product ID, load its image, preprocess, extract embeddings
    for pid in tqdm(prod_ids, desc="Embedding images"):
        # detect local file (we assume extension is .jpg/.png in data/images/)
        # We'll find exactly one file that starts with the ID
        folder = os.path.join("data", "images")
        candidates = [f for f in os.listdir(folder) if f.startswith(str(pid))]
        if not candidates:
            failed_ids.append(pid)
            # For missing image, push a zero‐vector or skip. Here we use zeros:
            embeddings.append(np.zeros(512, dtype=np.float32))
            continue

        img_path = os.path.join(folder, candidates[0])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Could not open image for id {pid} ({img_path}): {e}")
            failed_ids.append(pid)
            embeddings.append(np.zeros(512, dtype=np.float32))
            continue

        # Preprocess and forward‐pass
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)  # shape (1, 512)
            img_emb = outputs[0].cpu().numpy()            # convert to CPU numpy

        # Normalize to unit length
        norm = np.linalg.norm(img_emb)
        if norm > 0:
            img_emb = img_emb / norm

        embeddings.append(img_emb)

    # 4. Stack into a single array
    embedding_matrix = np.vstack(embeddings)  # shape: (N_products, 512)

    # 5. Save embeddings & index
    os.makedirs("models", exist_ok=True)
    npy_path = os.path.join("models", "catalog_embeddings.npy")
    idx_path = os.path.join("models", "catalog_index.json")

    np.save(npy_path, embedding_matrix)
    with open(idx_path, "w") as f:
        json.dump([int(x) for x in prod_ids], f)  # ensure IDs are JSON‐serializable

    print(f"Saved embeddings to {npy_path} (shape: {embedding_matrix.shape})")
    print(f"Saved index to {idx_path}")
    if failed_ids:
        print(f"[WARN] Failed to embed {len(failed_ids)} products: {failed_ids[:10]} ...")

    return embedding_matrix

if __name__ == "__main__":
    compute_clip_embeddings()
