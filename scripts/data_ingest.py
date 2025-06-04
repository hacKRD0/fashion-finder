# scripts/data_ingest.py

import os
import pandas as pd

def load_and_merge_catalog():
    """
    1) Reads product_data.xlsx and images.csv from data/
    2) Merges on 'id'
    3) Selects EXACTLY the first image_url per product 'id' as representative
    4) Saves a CSV of (id, title, product_type, rep_image_url) for downstream embedding.
    """
    # 1. Paths (assumes script run from project root)
    excel_path = os.path.join("data", "product_data.xlsx")
    images_csv_path = os.path.join("data", "images.csv")

    # 2. Load both files
    product_df = pd.read_excel(excel_path, engine="openpyxl")
    images_df = pd.read_csv(images_csv_path)

    # 3. Merge on 'id' (left‐join images onto products)
    merged = images_df.merge(
        product_df[["id", "title", "product_type"]],
        on="id",
        how="left",
        validate="many_to_one"  # multiple images → one product
    )

    # 4. For each 'id', pick the first image_url (to keep one representative)
    merged_sorted = merged.sort_values(by=["id"])  # ensure consistent ordering
    representative = (
        merged_sorted.groupby("id", as_index=False)
        .agg({
            "image_url": "first",      # take the first URL encountered
            "title": "first", 
            "product_type": "first"
        })
    )

    # 5. Save to CSV for easy reference
    out_path = os.path.join("data", "catalog_for_embedding.csv")
    representative.to_csv(out_path, index=False)
    print(f"Saved merged catalog → {out_path} (num products: {len(representative)})")

    return representative

if __name__ == "__main__":
    load_and_merge_catalog()
