import json
import os
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ZS_MODEL    = "facebook/bart-large-mnli"
VIBES_JSON  = os.path.join("data", "vibeslist.json")
TOP_K_VIBES = 3
# ─────────────────────────────────────────────────────────────────────────────

# Robust prompt definitions
VIBE_DEFINITIONS = """
You are a fashion-style vibe classification assistant.
Below are the definitions for each vibe label; use them to interpret the input caption and assign the top 1–3 vibes:

• Coquette: Whimsical, flirtatious details like bows, frills, and pastel romanticism.
• Clean Girl: Polished minimalism—neutral tones, “no-makeup” makeup, and sleek, effortless styling.
• Cottagecore: Rustic, nature-inspired simplicity—florals, lace, prairie silhouettes, and pastoral charm.
• Streetcore: Urban edge—graffiti graphics, techwear elements, chunky sneakers, and a youthful rebellious spirit.
• Y2K: Late-’90s/early-2000s futurism—metallics, velour tracksuits, bubblegum tones, and retro-tech accents.
• Boho: Free-spirited eclecticism—flowing dresses, embroidery, fringe, layered accessories, and earthy palettes.
• Party Glam: High-impact evening glamour—sequins, metallic fabrics, figure-hugging silhouettes, and bold accessories.
Instruction: Given the caption below, output a JSON array of the top 1–3 vibes (name + confidence score).
Caption:
"""

class VibeClassifier:
    def __init__(self):
        # Load the static list of vibes
        with open(VIBES_JSON, "r") as f:
            self.vibes = json.load(f)
        # Initialize zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=ZS_MODEL,
            device=-1  # CPU; set to 0 for GPU if available
        )

    def classify(self, text: str):
        """
        Given an input string (captions/hashtags), return up to TOP_K_VIBES vibes.
        """
        if not text or not text.strip():
            return []

        # Build the full prompt including definitions and the caption
        prompt = VIBE_DEFINITIONS + f"\"{text.strip()}\""

        # Call the zero-shot pipeline
        result = self.classifier(
            sequences=prompt,
            candidate_labels=self.vibes,
            multi_label=True  # allow multiple vibes
        )

        # Extract top-K labels and scores
        labels = result["labels"]
        scores = result["scores"]
        top = sorted(
            zip(labels, scores),
            key=lambda x: x[1],
            reverse=True
        )[:TOP_K_VIBES]

        return [{"vibe": label, "score": round(float(score), 4)} for label, score in top]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify a piece of text into fashion vibes (top 3)."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input caption or hashtags for vibe classification."
    )
    args = parser.parse_args()

    vc = VibeClassifier()
    vibes = vc.classify(args.text)
    print(json.dumps(vibes, indent=2))
