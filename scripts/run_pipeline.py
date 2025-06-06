import os
import json
import argparse

# Handle OpenMP conflicts if any
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from match_products import ProductMatcher
from classify_vibes import VibeClassifier

def run_end_to_end(video_path: str, captions: str, outputs_dir: str):
    """
    1. Instantiate ProductMatcher (loads + warms FAISS).
    2. Run detect_and_match(...) → matched products.
    3. Instantiate VibeClassifier → top 1–3 vibes.
    4. Assemble a JSON with both and write to outputs_dir/<video_id>.json.
    """
    video_id    = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(outputs_dir, exist_ok=True)
    output_json = os.path.join(outputs_dir, f"{video_id}.json")

    # A. Match products
    pm = ProductMatcher(device="cpu")
    matched_products = pm.detect_and_match(video_path, output_json_path=None)

    # B. Classify vibes
    vc = VibeClassifier()
    vibe_list = vc.classify(captions)

    # C. Assemble final JSON
    final = {
        "video_id": video_id,
        "vibes": vibe_list,           # e.g. [{"vibe":"Cottagecore","score":0.87}, ...]
        "products": matched_products  # list of matched product dicts
    }

    with open(output_json, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved end‐to‐end output → {output_json}")

    return final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: YOLO→CLIP→FAISS + Vibe Classification.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (MP4).")
    parser.add_argument(
        "--captions", type=str, default="",
        help="(Optional) Captions/hashtags/transcript text for vibe classification."
    )
    parser.add_argument(
        "--outputs", type=str, default="outputs",
        help="Directory to save final JSON (default: outputs/)."
    )
    args = parser.parse_args()

    run_end_to_end(args.video, args.captions, args.outputs)
