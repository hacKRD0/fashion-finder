# scripts/compute_embeddings.py

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def load_clip_model(device: str = None) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load and return the CLIP model and processor.
    
    Args:
        device: Device to load the model on ('cuda' or 'cpu'). If None, auto-detects.
        
    Returns:
        tuple: (model, processor) - Loaded CLIP model and processor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP model on device: {device}")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def process_single_image(
    image_path: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str
) -> Optional[np.ndarray]:
    """Process a single image and return its embedding.
    
    Args:
        image_path: Path to the image file
        model: Loaded CLIP model
        processor: CLIP processor
        device: Device to run the model on
        
    Returns:
        Optional[np.ndarray]: Normalized image embedding or None if processing fails
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            img_emb = outputs[0].cpu().numpy()
            
        # Normalize to unit length
        norm = np.linalg.norm(img_emb)
        return img_emb / norm if norm > 0 else None
        
    except Exception as e:
        print(f"[ERROR] Could not process image {image_path}: {e}")
        return None


def process_product_images(
    product_folder: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str
) -> Optional[np.ndarray]:
    """Process all images for a single product and return the centroid embedding.
    
    Args:
        product_folder: Path to the folder containing product images
        model: Loaded CLIP model
        processor: CLIP processor
        device: Device to run the model on
        
    Returns:
        Optional[np.ndarray]: Centroid embedding of all product images or None if no valid images
    """
    if not os.path.exists(product_folder):
        return None
        
    embeddings = []
    for img_file in os.listdir(product_folder):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(product_folder, img_file)
        emb = process_single_image(img_path, model, processor, device)
        if emb is not None:
            embeddings.append(emb)
    
    return np.mean(embeddings, axis=0) if embeddings else None


def save_embeddings(
    embeddings: np.ndarray,
    product_ids: List[int],
    output_dir: str = "models"
) -> Tuple[str, str]:
    """Save embeddings and product IDs to disk.
    
    Args:
        embeddings: Numpy array of embeddings
        product_ids: List of product IDs corresponding to the embeddings
        output_dir: Directory to save the output files
        
    Returns:
        tuple: Paths to the saved embeddings and index files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy array if not already
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    # Ensure product IDs are JSON-serializable
    product_ids = [int(pid) for pid in product_ids]
    
    # Save embeddings and index
    embeddings_path = os.path.join(output_dir, "catalog_embeddings.npy")
    index_path = os.path.join(output_dir, "catalog_index.json")
    
    np.save(embeddings_path, embeddings)
    with open(index_path, "w") as f:
        json.dump(product_ids, f)
    
    return embeddings_path, index_path


def compute_clip_embeddings(
    input_dir: str = "data/processed",
    catalog_path: str = "data/catalog_for_embedding.csv",
    output_dir: str = "models",
    device: str = None,
    batch_size: int = 32,
    limit: int = None
) -> Tuple[np.ndarray, List[int]]:
    """Compute CLIP embeddings for all product images and save the results.
    
    Args:
        input_dir: Directory containing processed product images
        catalog_path: Path to the catalog CSV file
        output_dir: Directory to save the output files
        device: Device to run the model on (None for auto-detect)
        batch_size: Batch size for processing (not used in current implementation)
        limit: Maximum number of products to process (for testing)
        
    Returns:
        tuple: (embeddings, product_ids) - Numpy array of embeddings and corresponding product IDs
    """
    # Load model and processor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_clip_model(device)
    
    # Load product IDs from catalog
    df = pd.read_csv(catalog_path)
    all_product_ids = df["id"].tolist()
    
    if limit:
        all_product_ids = all_product_ids[:limit]
    
    # Process each product
    embeddings = []
    valid_product_ids = []
    
    for pid in tqdm(all_product_ids, desc="Processing products"):
        product_folder = os.path.join(input_dir, str(pid))
        centroid = process_product_images(product_folder, model, processor, device)
        
        if centroid is not None:
            embeddings.append(centroid)
            valid_product_ids.append(pid)
    
    # Convert to numpy array
    if not embeddings:
        raise ValueError("No valid embeddings were generated")
        
    embeddings_array = np.vstack(embeddings)
    
    # Save results
    emb_path, idx_path = save_embeddings(embeddings_array, valid_product_ids, output_dir)
    
    print(f"\nSuccessfully processed {len(valid_product_ids)} products")
    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved index to: {idx_path}")
    
    return embeddings_array, valid_product_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute CLIP embeddings for product images')
    parser.add_argument('--input', '-i', default='data/processed',
                        help='Directory containing processed product images')
    parser.add_argument('--catalog', '-c', default='data/catalog_for_embedding.csv',
                        help='Path to catalog CSV file')
    parser.add_argument('--output', '-o', default='models',
                        help='Output directory for saving embeddings')
    parser.add_argument('--device', '-d', default=None,
                        help='Device to use (cuda/cpu), auto-detects if not specified')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of products to process (for testing)')
    
    args = parser.parse_args()
    
    compute_clip_embeddings(
        input_dir=args.input,
        catalog_path=args.catalog,
        output_dir=args.output,
        device=args.device,
        limit=args.limit
    )
