import json
import os
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Config (if you want to swap models)
# ─────────────────────────────────────────────────────────────────────────────
ZS_MODEL    = "facebook/bart-large-mnli"
VIBES_JSON  = os.path.join("data", "vibeslist.json")
TOP_K_VIBES = 3
# ─────────────────────────────────────────────────────────────────────────────

class VibeClassifier:
    def __init__(self):
        # Load the static list of 7 vibes
        with open(VIBES_JSON, "r") as f:
            self.vibes = json.load(f)
        # Initialize zero-shot classification pipeline
        self.classifier = pipeline("zero-shot-classification", model=ZS_MODEL)

    def classify(self, text: str):
        """
        Given an input string (captions/hashtags), return up to TOP_K_VIBES vibes.
        """
        if not text or not text.strip():
            return []

        result = self.classifier(text, candidate_labels=self.vibes)
        top_labels = result["labels"][:TOP_K_VIBES]
        top_scores = result["scores"][:TOP_K_VIBES]
        return [{"vibe": vib, "score": round(float(sc), 4)} for vib, sc in zip(top_labels, top_scores)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a piece of text into 7 vibes (top 3).")
    parser.add_argument("--text", type=str, required=True, help="Input text (captions/hashtags).")
    args = parser.parse_args()

    vc = VibeClassifier()
    vibes = vc.classify(args.text)
    print(f"Input text: {args.text}")
    print("Predicted vibes:")
    for item in vibes:
        print(f"  – {item['vibe']} (score={item['score']})")
