# scripts/download_images.py

import os
import ast
import argparse
import requests
import pandas as pd
from urllib.parse import urlparse

def download_catalog_images(limit=None):
    """
    Reads data/catalog_for_embedding.csv, and for each row:
    - downloads all image_urls for the product
    - saves them in a subfolder named after the product id
    
    Args:
        limit (int, optional): Maximum number of products to process. If None, processes all products.
    """
    catalog_path = os.path.join("data", "catalog_for_embedding.csv")
    df = pd.read_csv(catalog_path)
    
    # Apply limit if specified
    if limit is not None:
        df = df.head(limit)
        print(f"Limiting to first {limit} products")

    base_folder = os.path.join("data", "images")
    os.makedirs(base_folder, exist_ok=True)
    
    total_products = len(df)
    print(f"Processing {total_products} products...")

    for idx, row in df.iterrows():
        prod_id = row["id"]
        urls_str = row["image_url"]
        
        if pd.isna(urls_str) or not urls_str.strip():
            print(f"[WARN] id {prod_id} has no image_urls → skipping.")
            continue
            
        try:
            # Safely evaluate the string as a Python list
            urls = ast.literal_eval(urls_str)
            if not isinstance(urls, list):
                urls = [urls]  # If it's a single URL, make it a list
        except (ValueError, SyntaxError) as e:
            print(f"[WARN] Could not parse URLs for id {prod_id}: {e}")
            continue

        # Create product subfolder
        prod_folder = os.path.join(base_folder, str(prod_id))
        os.makedirs(prod_folder, exist_ok=True)

        # Download each image
        for i, url in enumerate(urls):
            if pd.isna(url):
                continue

            # Determine extension from URL
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            ext = os.path.splitext(filename)[1] or ".jpg"

            # Add sequence number to filename
            local_fname = f"{i:02d}{ext}"
            save_path = os.path.join(prod_folder, local_fname)

            # Skip download if file already exists
            if os.path.exists(save_path):
                continue

            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                print(f"Downloaded {prod_id}/{i:02d} → {local_fname}")
            except Exception as e:
                print(f"[ERROR] Could not download {url} for id {prod_id}: {e}")

    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download product images from catalog')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of products to process (default: all)')
    args = parser.parse_args()
    
    download_catalog_images(limit=args.limit)
