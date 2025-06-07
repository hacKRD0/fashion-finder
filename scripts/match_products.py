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
import time

# ─────────────────────────────────────────────────────────────────────────────
# Configurable parameters
# ─────────────────────────────────────────────────────────────────────────────

# Paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure model directory exists
YOLO_WEIGHTS = "yolov8-fashion.pt"  # Custom fashion-trained weights
FAISS_INDEX = os.path.join(MODEL_DIR, "catalog_faiss.index")
FAISS_INDEX_JSON = os.path.join(MODEL_DIR, "catalog_index.json")

# YOLO settings
YOLO_BATCH_SIZE = 4               # frames per YOLO batch call
CONF_THRESH = 0.25                # YOLO confidence threshold
IOU_THRESH = 0.45                 # YOLO NMS IoU threshold
FRAME_SKIP = 5                     # process every 5th frame to speed up processing

# CLIP settings
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE = 8               # crops per CLIP batch call

# Class mapping for YOLO model
CLASS_MAP = {
    "accessories": 0,
    "bags": 1,
    "clothing": 2,
    "shoes": 3
}

# Target classes to process (using class indices)
TARGET_CLASSES = [CLASS_MAP["clothing"]]  # Default to only clothing

# Matching
TOP_K = 5                         # neighbors per query
SIMILARITY_EXACT = 0.90
SIMILARITY_SIMILAR = 0.75

# Debugging
SAVE_CROPS = True                # Set to True to save cropped images
CROPS_OUTPUT_DIR = "debug_crops"  # Directory to save cropped images

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def filter_detections(detections, target_classes=None):
    """Filter detections to only include target classes.
    
    Args:
        detections: List of YOLO Results objects
        target_classes: List of class indices to include (None for all)
        
    Returns:
        Filtered list of Results objects with only target class detections
    """
    if target_classes is None:
        return detections
        
    filtered = []
    for det in detections:
        if det.boxes is None or len(det.boxes) == 0:
            filtered.append(None)
            continue
            
        # Get class IDs from the detection
        class_ids = det.boxes.cls.cpu().numpy().astype(int)
        # Filter boxes that match target classes
        mask = np.isin(class_ids, target_classes)
        
        if not np.any(mask):
            filtered.append(None)
            continue
            
        # Create a new Results object with filtered boxes
        new_det = det.new()
        # Copy original attributes
        for k, v in det.__dict__.items():
            if k != 'boxes':
                setattr(new_det, k, v)
        
        # Filter and set boxes
        new_det.boxes = det.boxes[mask]
        filtered.append(new_det)
    
    return filtered

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
        if not os.path.exists(YOLO_WEIGHTS):
            print(f"Downloading YOLO model to {YOLO_WEIGHTS}...")
            try:
                # Try to download the model if not found
                from ultralytics.hub.utils import safe_download
                url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"  # Replace with actual model URL if different
                safe_download(url, YOLO_WEIGHTS)
                print("YOLO model downloaded successfully!")
            except Exception as e:
                raise FileNotFoundError(
                    f"YOLO model not found at {YOLO_WEIGHTS} and could not be downloaded. "
                    f"Please download it manually and place it in the {MODEL_DIR} directory."
                ) from e
                
        self.yolo = YOLO(YOLO_WEIGHTS)

    def _save_detection_crops(self, frames, detections, frame_indices):
        """
        Save cropped images of detections for debugging purposes.
        
        Args:
            frames: List of frames (numpy arrays in BGR format)
            detections: List of YOLO Results objects (one per frame)
            frame_indices: List of original frame indices
        """
        if not SAVE_CROPS:
            return
            
        os.makedirs(CROPS_OUTPUT_DIR, exist_ok=True)
        
        for i, (frame, det) in enumerate(zip(frames, detections)):
            if det is None or det.boxes is None:
                continue
                
            frame_idx = frame_indices[i] if i < len(frame_indices) else i
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for box in det.boxes:
                try:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Skip invalid boxes
                    if x1 >= x2 or y1 >= y2:
                        continue
                        
                    # Extract and save crop
                    crop = frame_rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    # Create a unique filename
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    timestamp = int(time.time() * 1000)
                    crop_filename = f"frame_{frame_idx:06d}_class_{class_id}_conf_{conf:.2f}_{timestamp}.jpg"
                    crop_path = os.path.join(CROPS_OUTPUT_DIR, crop_filename)
                    
                    # Save the cropped image
                    cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    
                except Exception as e:
                    print(f"Error saving crop: {e}")
                    continue

    def _batch_clip_embed(self, frames, detections):
        """
        Extract object crops from frames based on detections and generate CLIP embeddings.
        
        Args:
            frames: List of frames (numpy arrays in BGR format)
            detections: List of YOLO Results objects (one per frame)
            
        Returns:
            List of dicts containing frame_idx, bbox, and embedding for each detection
        """
        all_embeddings = []
        
        # Process frames and their detections in batches
        for i in range(0, len(frames), CLIP_BATCH_SIZE):
            batch_frames = frames[i:i + CLIP_BATCH_SIZE]
            batch_dets = detections[i:i + CLIP_BATCH_SIZE]
            
            crops = []
            crop_metadata = []  # Store metadata for each crop
            
            # Extract crops from frames based on detections
            for j, (frame, det) in enumerate(zip(batch_frames, batch_dets)):
                if det is None or det.boxes is None or len(det.boxes) == 0:
                    continue
                    
                # Convert frame to RGB for CLIP
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process each detection in the frame
                for box in det.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Skip invalid boxes
                    if x1 >= x2 or y1 >= y2:
                        continue
                        
                    # Extract crop
                    crop = frame_rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    # Resize crop to minimum size expected by CLIP
                    min_size = 224
                    h, w = crop.shape[:2]
                    if h < min_size or w < min_size:
                        scale = max(min_size / h, min_size / w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Convert to PIL Image for CLIP processor
                    try:
                        crop_pil = Image.fromarray(crop)
                        crops.append(crop_pil)
                        crop_metadata.append({
                            'frame_idx': i,
                            'bbox': [int(x) for x in [x1, y1, x2, y2]],
                            'class_id': int(box.cls[0]),
                            'conf': float(box.conf[0])
                        })
                    except Exception as e:
                        print(f"Error processing crop: {e}")
                        continue
            
            if not crops:
                continue
                
            # Process crops with CLIP
            try:
                inputs = self.clip_processor(
                    images=crops, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    # Get image features
                    outputs = self.clip_model.get_image_features(**inputs)
                    
                    # Normalize embeddings
                    image_embeds = outputs / outputs.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy and store with metadata
                    for idx, meta in enumerate(crop_metadata):
                        all_embeddings.append({
                            'frame_idx': meta['frame_idx'],
                            'bbox': meta['bbox'],
                            'class_id': meta['class_id'],
                            'conf': meta['conf'],
                            'embedding': image_embeds[idx].cpu().numpy().astype(np.float32)
                        })
                        
            except Exception as e:
                print(f"Error processing CLIP batch: {e}")
                continue
        
        return all_embeddings

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

    def detect_and_match(self, video_path, output_json_path=None, max_frames=None, target_classes=None):
        """
        Process a video to detect fashion items and match them with the catalog.
        
        Args:
            video_path: Path to the input video file
            output_json_path: Optional path to save results as JSON
            max_frames: Maximum number of frames to process (for testing)
            target_classes: List of class indices to process (None for default TARGET_CLASSES)
            
        Returns:
            List of matched products with metadata
        """
        if target_classes is None:
            target_classes = TARGET_CLASSES
            
        # Convert class names to indices if strings are provided
        if target_classes and isinstance(target_classes[0], str):
            target_classes = [CLASS_MAP.get(cls, -1) for cls in target_classes]
            target_classes = [idx for idx in target_classes if idx != -1]  # Remove invalid classes
        
        if not target_classes:
            raise ValueError("No valid target classes provided")
            
        print(f"Processing video: {video_path}")
        print(f"Target classes: {[cls for cls, idx in CLASS_MAP.items() if idx in target_classes]}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames:
            frame_count = min(frame_count, max_frames)
            
        print(f"Total frames: {frame_count}, FPS: {fps:.2f}")
        
        best_matches = {}  # pid → {similarity, match_type, bbox, frame_no, yolo_conf, yolo_class_id}
        frame_batch = []
        frame_indices = []
        
        pbar = tqdm(total=frame_count, desc="Processing frames")
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_idx >= max_frames):
                    break
                    
                # Process every FRAME_SKIP-th frame
                if frame_idx % FRAME_SKIP == 0:
                    frame_batch.append(frame.copy())
                    frame_indices.append(frame_idx)
                    
                    # Process batch when we have enough frames
                    if len(frame_batch) == YOLO_BATCH_SIZE:
                        self._process_batch(frame_batch, frame_indices, target_classes, best_matches)
                        frame_batch = []
                        frame_indices = []
                
                frame_idx += 1
                pbar.update(1)
                
            # Process any remaining frames
            if frame_batch:
                self._process_batch(frame_batch, frame_indices, target_classes, best_matches)
                
        finally:
            cap.release()
            pbar.close()
        
        # Convert best_matches dict to list of dicts
        matched_list = []
        for pid, info in best_matches.items():
            matched_list.append({
                "product_id": int(pid),
                "match_type": info["match_type"],
                "similarity": float(info["similarity"]),
                "frame_no": int(info["frame_no"]),
                "yolo_confidence": float(info["yolo_conf"]),
                "yolo_class_id": int(info["yolo_class_id"]),
                "bbox": [int(x) for x in info["bbox"]]
            })
        
        # Sort by match type (exact > similar) then by similarity (descending)
        matched_list.sort(
            key=lambda x: (0 if x["match_type"] == "exact" else 1, -x["similarity"])
        )
        
        # Save results if output path is provided
        if output_json_path:
            output_dir = os.path.dirname(output_json_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(output_json_path, 'w') as f:
                json.dump(matched_list, f, indent=2)
                
            print(f"Saved {len(matched_list)} matched products to {output_json_path}")
        
        return matched_list
        
    def _process_batch(self, frames, frame_indices, target_classes, best_matches):
        """Process a batch of frames with YOLO and CLIP."""
        # Convert frames to RGB for YOLO
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        
        # Run YOLO on the batch
        with torch.no_grad():
            results = self.yolo(rgb_frames, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
        
        # Filter detections to only include target classes
        filtered_results = filter_detections(results, target_classes)
        
        # Save detection crops if enabled (for debugging/validation)
        if SAVE_CROPS:
            self._save_detection_crops(frames, filtered_results, frame_indices)
        
        # Process detections with CLIP and update best matches
        embeddings = self._batch_clip_embed(frames, filtered_results)
        
        # Process each embedding and update best matches
        for emb_info in embeddings:
            if emb_info is None:
                continue
                
            frame_idx = emb_info['frame_idx']
            bbox = emb_info['bbox']
            embedding = emb_info['embedding']
            
            # Skip if no detections for this frame
            if filtered_results[frame_idx] is None or filtered_results[frame_idx].boxes is None:
                continue
            
            # Find matching detections for this embedding
            for box in filtered_results[frame_idx].boxes:
                det_bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
                if det_bbox == bbox:  # Match found
                    # Search in FAISS index
                    D, I = self.index.search(np.expand_dims(embedding, 0), TOP_K)
                    
                    # Process matches
                    for score, idx in zip(D[0], I[0]):
                        if idx < 0 or idx >= len(self.prod_ids):
                            continue
                            
                        # Determine match type
                        if score >= SIMILARITY_EXACT:
                            match_type = "exact"
                        elif score >= SIMILARITY_SIMILAR:
                            match_type = "similar"
                        else:
                            continue
                            
                        # Get detection metadata
                        pid = int(self.prod_ids[idx])
                        det_conf = float(box.conf[0])
                        det_cls = int(box.cls[0])
                        
                        # Update best matches
                        prev = best_matches.get(pid)
                        if prev is None or score > prev["similarity"]:
                            best_matches[pid] = {
                                "similarity": float(score),
                                "match_type": match_type,
                                "bbox": bbox,
                                "frame_no": frame_indices[frame_idx],
                                "yolo_conf": det_conf,
                                "yolo_class_id": det_cls
                            }
                    break  # Move to next embedding


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
