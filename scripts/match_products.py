import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import faiss

# ─────────────────────────────────────────────────────────────────────────────
# Configurable parameters
# ─────────────────────────────────────────────────────────────────────────────

YOLO_WEIGHTS      = "yolov8n.pt"    # or your custom fashion‐trained weights
CONF_THRESH       = 0.25            # YOLO confidence threshold
IOU_THRESH        = 0.45            # YOLO NMS IoU threshold
FRAME_SKIP        = 5               # process every 5th frame

YOLO_BATCH_SIZE   = 4               # frames per YOLO batch call
CLIP_BATCH_SIZE   = 8               # crops per CLIP batch call

CLIP_MODEL        = "openai/clip-vit-base-patch32"
TOP_K             = 5               # neighbors per query
SIMILARITY_EXACT  = 0.90
SIMILARITY_SIMILAR = 0.75

# ─────────────────────────────────────────────────────────────────────────────

class ProductMatcher:
    def __init__(self, device="cpu"):
        self.device = device

        # 1. Load & warm FAISS index + product IDs
        faiss_path = os.path.join("models", "catalog_faiss.index")
        idx_json   = os.path.join("models", "catalog_index.json")
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"{faiss_path} not found; run build_faiss_index.py first.")

        # Load FAISS index
        self.index = faiss.read_index(faiss_path)
        # Warm up (dummy search) to load data into CPU caches
        dim = self.index.d
        zero_vec = np.zeros((1, dim), dtype=np.float32)
        _scores, _idxs = self.index.search(zero_vec, 1)

        # Load product ID list in same order as embeddings
        with open(idx_json, "r") as f:
            self.prod_ids = json.load(f)  # list of ints

        # 2. Load CLIP model & processor
        self.clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

        # 3. Load YOLOv8 model (CPU)
        self.yolo = YOLO(YOLO_WEIGHTS)

    def _batch_clip_embed(self, pil_images):
        """
        Convert a list of PIL images into normalized 512‐dim CLIP embeddings, batched.
        """
        embeddings = []
        for i in range(0, len(pil_images), CLIP_BATCH_SIZE):
            batch = pil_images[i : i + CLIP_BATCH_SIZE]
            inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)  # (B,512)
            np_feats = feats.cpu().numpy().astype(np.float32)        # (B,512)
            norms = np.linalg.norm(np_feats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            np_feats = np_feats / norms
            embeddings.extend(np_feats)
        return embeddings  # list of (512,) arrays

    def _match_embedding(self, emb):
        """
        Query FAISS index with one (512,) normalized embedding. Return top K (product_id, similarity).
        """
        D, I = self.index.search(np.expand_dims(emb, axis=0), TOP_K)
        neighbors = []
        for sim, idx in zip(D[0], I[0]):
            pid = self.prod_ids[idx]
            neighbors.append((pid, float(sim)))
        return neighbors

    def detect_and_match(self, video_path, output_json_path=None, max_frames=None):
        """
        1. Uses FRAME_SKIP and batches YOLO calls (YOLO_BATCH_SIZE).
        2. Batches CLIP crops (CLIP_BATCH_SIZE).
        3. Aggregates best match per product ID (exact/similar).
        4. Optionally writes JSON and returns matched_list.
        """
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_no     = 0
        frames_buf   = []   # store frames for batching
        buf_framenos = []   # store corresponding frame indices

        best_matches = {}   # pid → {similarity, match_type, bbox, frame_no, yolo_conf, yolo_class_id}

        pbar = tqdm(
            total=frame_count if max_frames is None else min(frame_count, max_frames),
            desc=f"Detecting/matching {os.path.basename(video_path)}"
        )

        def process_yolo_batch(frames_batch, framenos_batch):
            """
            Run YOLO on the batch of frames, crop detections, batch‐embed via CLIP, and update best_matches.
            """
            # 1. Convert BGR→RGB
            batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_batch]
            # 2. Run YOLO on list of frames
            results = self.yolo(batch_rgb, imgsz=640, conf=CONF_THRESH, iou=IOU_THRESH)
            crops = []
            meta  = []

            # 3. Collect all crops + metadata
            for batch_idx, res in enumerate(results):
                dets = res.boxes
                frm_no = framenos_batch[batch_idx]
                orig = frames_batch[batch_idx]

                for box, conf, cls in zip(dets.xyxy.cpu().numpy(),
                                          dets.conf.cpu().numpy(),
                                          dets.cls.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int).tolist()
                    crop_bgr = orig[y1:y2, x1:x2]
                    if crop_bgr.size == 0:
                        continue
                    pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                    crops.append(pil_crop)
                    meta.append({
                        "frame_no": frm_no,
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf),
                        "cls": int(cls)
                    })

            if not crops:
                return

            # 4. Batch‐embed all crops
            embeddings = self._batch_clip_embed(crops)

            # 5. FAISS search & update best_matches
            for emb_vec, m in zip(embeddings, meta):
                neighbors = self._match_embedding(emb_vec)
                for pid, sim in neighbors:
                    if sim >= SIMILARITY_EXACT:
                        mtype = "exact"
                    elif sim >= SIMILARITY_SIMILAR:
                        mtype = "similar"
                    else:
                        continue
                    prev = best_matches.get(pid)
                    if prev is None or sim > prev["similarity"]:
                        best_matches[pid] = {
                            "similarity": sim,
                            "match_type": mtype,
                            "bbox": m["bbox"],
                            "frame_no": m["frame_no"],
                            "yolo_conf": m["conf"],
                            "yolo_class_id": m["cls"]
                        }

        # Main loop: read frames, apply FRAME_SKIP, buffer for YOLO_BATCH_SIZE, then process
        while True:
            ret, frame = vidcap.read()
            if not ret:
                break

            if frame_no % FRAME_SKIP == 0:
                frames_buf.append(frame.copy())
                buf_framenos.append(frame_no)
                if len(frames_buf) == YOLO_BATCH_SIZE:
                    process_yolo_batch(frames_buf, buf_framenos)
                    frames_buf.clear()
                    buf_framenos.clear()

            frame_no += 1
            pbar.update(1)
            if max_frames is not None and frame_no >= max_frames:
                break

        # Process any leftover buffered frames
        if frames_buf:
            process_yolo_batch(frames_buf, buf_framenos)

        pbar.close()
        vidcap.release()

        # Convert best_matches dict → list of dicts
        matched_list = []
        for pid, info in best_matches.items():
            matched_list.append({
                "product_id": int(pid),
                "match_type": info["match_type"],
                "confidence": round(info["similarity"], 4),
                "bbox": info["bbox"],
                "frame_no": info["frame_no"],
                "yolo_class_id": info["yolo_class_id"],
                "yolo_conf": round(info["yolo_conf"], 4),
            })

        # Optionally save JSON
        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            out_dict = {
                "video_id": os.path.splitext(os.path.basename(video_path))[0],
                "products": matched_list
            }
            with open(output_json_path, "w") as f:
                json.dump(out_dict, f, indent=2)
            print(f"Saved product matches → {output_json_path}")

        return matched_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO → CLIP → FAISS matching on a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4 video")
    parser.add_argument(
        "--out_json", type=str, default=None,
        help="If provided, save a JSON with matched products (without vibes)."
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="If you want to limit the number of frames processed (for faster testing)."
    )
    args = parser.parse_args()

    pm = ProductMatcher(device="cpu")
    pm.detect_and_match(args.video, output_json_path=args.out_json, max_frames=args.max_frames)
