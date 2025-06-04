# scripts/download_images.py

import os
import requests
import pandas as pd
from urllib.parse import urlparse

def download_catalog_images():
    """
    Reads data/catalog_for_embedding.csv, and for each row:
    - downloads the image_url
    - saves it as data/images/{id}.{ext}
    """
    catalog_path = os.path.join("data", "catalog_for_embedding.csv")
    df = pd.read_csv(catalog_path)

    images_folder = os.path.join("data", "images")
    os.makedirs(images_folder, exist_ok=True)

    for idx, row in df.iterrows():
        prod_id = row["id"]
        url = row["image_url"]
        if pd.isna(url):
            print(f"[WARN] id {prod_id} has no image_url → skipping.")
            continue

        # Determine extension from URL (e.g., .jpg or .png)
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)        # e.g. "some_img.jpg"
        ext = os.path.splitext(filename)[1] or ".jpg"  # fallback to .jpg

        local_fname = f"{prod_id}{ext}"
        save_path = os.path.join(images_folder, local_fname)

        # Skip download if file already exists
        if os.path.exists(save_path):
            continue

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(resp.content)
            print(f"Downloaded {prod_id} → {local_fname}")
        except Exception as e:
            print(f"[ERROR] Could not download {url} for id {prod_id}: {e}")

    print("Download complete.")

if __name__ == "__main__":
    download_catalog_images()
