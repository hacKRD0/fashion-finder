# scripts/detect_objects.py

import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO

# Class mapping for YOLO model
CLASS_MAP = {
    "accessories": 0,
    "bags": 1,
    "clothing": 2,
    "shoes": 3
}

# List of target classes to include in the output (using class indices)
TARGET_CLASSES = [CLASS_MAP["clothing"]]  # Default to only clothing

def detect_and_crop_clothing(
    input_folder: str,
    output_folder: str,
    model_path: str = "yolov8-fashion.pt",
    target_classes: List[int] = None,
    class_map: Dict[str, int] = None,
    max_products: int = None
) -> None:
    """
    Detects fashion items in images using YOLOv8 and saves cropped versions.
    
    Args:
        input_folder: Path to the folder containing product images
        output_folder: Path where cropped images will be saved
        model_path: Path to the YOLOv8 model weights
        target_classes: List of class indices to include in the output
        class_map: Dictionary mapping class names to their indices
        max_products: Maximum number of product folders to process (None for no limit)
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES
    if class_map is None:
        class_map = CLASS_MAP
        
    # Reverse class map for looking up names
    idx_to_class = {v: k for k, v in class_map.items()}
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Process each product folder
    processed_count = 0
    for prod_id in os.listdir(input_folder):
        # Check if we've reached the maximum number of products to process
        if max_products is not None and processed_count >= max_products:
            print(f"Reached maximum limit of {max_products} products. Stopping processing.")
            break
            
        processed_count += 1
        prod_path = os.path.join(input_folder, prod_id)
        if not os.path.isdir(prod_path):
            continue
            
        # Create product subfolder in output
        prod_output = os.path.join(output_folder, prod_id)
        os.makedirs(prod_output, exist_ok=True)
        
        # Process each image in the product folder
        for img_file in os.listdir(prod_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(prod_path, img_file)
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not load image: {img_path}")
                    continue
                    
                # Run detection
                results = model(img)
                
                # Process each detection
                for i, r in enumerate(results):
                    boxes = r.boxes
                    for box in boxes:
                        # Get class and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Skip if not in target classes
                        if class_id not in target_classes:
                            continue
                            
                        # Get class name
                        class_name = idx_to_class.get(class_id, f"class_{class_id}")
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Skip if the box is too small
                        if (x2 - x1) < 10 or (y2 - y1) < 10:
                            continue
                            
                        # Crop the item
                        cropped = img[y1:y2, x1:x2]
                        
                        # Save cropped image with class and confidence in the filename
                        base_name = Path(img_file).stem
                        cropped_path = os.path.join(
                            prod_output,
                            f"{base_name}_{class_name}_{confidence:.2f}_crop_{i}{Path(img_file).suffix}"
                        )
                        cv2.imwrite(cropped_path, cropped)
                        print(f"Saved {class_name} (conf: {confidence:.2f}): {cropped_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect and crop fashion items from product images')
    parser.add_argument('--input', '-i', default='data/images',
                        help='Input folder containing product images (default: data/images)')
    parser.add_argument('--output', '-o', default='data/processed',
                        help='Output folder for cropped images (default: data/processed)')
    parser.add_argument('--model', '-m', default='yolov8-fashion.pt',
                        help='Path to YOLOv8 model weights (default: yolov8-fashion.pt)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Maximum number of products to process (default: no limit)')
    parser.add_argument('--classes', '-c', nargs='+', default=['clothing'],
                        help='List of classes to detect (default: clothing)')
    
    args = parser.parse_args()
    
    # Convert class names to indices
    target_classes = []
    for cls in args.classes:
        if cls not in CLASS_MAP:
            print(f"Warning: Unknown class '{cls}'. Available classes: {list(CLASS_MAP.keys())}")
            continue
        target_classes.append(CLASS_MAP[cls])
    
    if not target_classes:
        print("No valid target classes specified. Using default (clothing).")
        target_classes = [CLASS_MAP["clothing"]]
    
    print(f"Processing up to {args.limit if args.limit else 'all'} products")
    print(f"Target classes: {[cls for cls, idx in CLASS_MAP.items() if idx in target_classes]}")
    
    detect_and_crop_clothing(
        input_folder=args.input,
        output_folder=args.output,
        model_path=args.model,
        target_classes=target_classes,
        max_products=args.limit
    )
