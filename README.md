# Fashion Finder

[![Loom Demo](https://img.shields.io/badge/View-Demo-552586?style=for-the-badge&logo=loom&logoColor=white)](https://www.loom.com/share/106131bf2e3a4c74ad42ad622bbd48a4?sid=1b8eb8c4-75eb-487a-aa9d-aa8342c7bd67)

A computer vision pipeline for detecting fashion items in videos, matching them with a product catalog, and classifying the overall style vibes.

## Project Overview

This project implements an end-to-end pipeline for processing fashion content, including:
- **Object Detection**: Detect fashion items in videos using YOLOv8
- **Product Matching**: Match detected items with a product catalog using CLIP and FAISS
- **Vibe Classification**: Classify the overall style of the content using zero-shot classification
- **Debugging Tools**: Save debug outputs including cropped detections and matched products

## Features

- **Video Processing**: Process video files to detect fashion items frame by frame
- **Product Matching**: Match detected items with a product catalog using visual similarity
- **Style Classification**: Classify the overall style/vibe of the content
- **Batch Processing**: Process multiple videos in a directory
- **Debug Outputs**: Save intermediate results for analysis and debugging

## Models Used

### 1. Object Detection
- **Model**: [YOLOv8n-clothing-detection](https://huggingface.co/kesimeg/yolov8n-clothing-detection)
  - **Version**: Fine-tuned YOLOv8n for fashion items
  - **Classes**: Clothing, bags, shoes, and accessories
  - **Task**: Detects fashion items in videos
  - **Input**: Video frames
  - **Output**: Bounding boxes around fashion items with class labels

### 2. Product Matching
- **Model**: CLIP (Contrastive Language-Image Pretraining)
  - **Version**: `openai/clip-vit-base-patch32`
  - **Task**: Generates embeddings for visual similarity search
  - **Input**: Cropped fashion item images
  - **Output**: Feature vectors for product matching

### 3. Vibe Classification
- **Model**: BART (Bidirectional and Auto-Regressive Transformers)
  - **Version**: `facebook/bart-large-mnli`
  - **Task**: Zero-shot text classification for fashion vibes
  - **Input**: Text descriptions or hashtags
  - **Output**: Top 1-3 fashion vibes with confidence scores

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-finder.git
   cd fashion-finder
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Full Pipeline

Process a single video with captions:
```bash
# Basic usage with captions provided directly
python scripts/run_pipeline.py \
    --video path/to/your/video.mp4 \
    --captions "Your video description or hashtags here" \
    --outputs outputs/

# With captions from a file
python scripts/run_pipeline.py \
    --video path/to/your/video.mp4 \
    --captions "$(<path/to/captions.txt)" \
    --outputs outputs/

# Limit processing to first 100 frames (for testing)
python scripts/run_pipeline.py \
    --video path/to/your/video.mp4 \
    --captions "Your description" \
    --outputs outputs/ \
    --max_frames 100
```

Process all videos in a directory (will look for .mp4 files and matching .txt files for captions):
```bash
# Process all .mp4 files in directory (looks for matching .txt files)
python scripts/run_pipeline.py --video path/to/videos/ --outputs outputs/

# Process specific file types with custom extensions
find path/to/videos/ -name "*.mp4" | xargs -I {} python scripts/run_pipeline.py --video {} --outputs outputs/
```

### 2. Individual Components

#### Product Matching Only
```bash
# Basic product matching
python scripts/match_products.py --video path/to/your/video.mp4 --out_json output.json

# With custom confidence threshold
python scripts/match_products.py \
    --video path/to/your/video.mp4 \
    --out_json output.json \
    --conf_thresh 0.6

# Process specific clothing categories only
python scripts/match_products.py \
    --video path/to/your/video.mp4 \
    --out_json output.json \
    --classes clothing shoes
```

#### Vibe Classification Only
```bash
# Basic vibe classification
python scripts/classify_vibes.py --text "Your fashion description or hashtags here"

# Classify text from a file
python scripts/classify_vibes.py --text "$(<path/to/description.txt)"

# Save output to a file
python scripts/classify_vibes.py --text "Your text" > vibe_results.json
```

### 3. Debugging and Visualization

View debug outputs:
```bash
# View debug crops (saved in debug_crops/)
ls -l debug_crops/*/

# View matched product images (saved in debug_outputs/)
ls -l debug_outputs/*/

# View JSON results
cat outputs/your_video_name.json | jq .  # requires jq for pretty printing
```

## Project Structure

```
├── data/
│   ├── vibeslist.json       # List of fashion vibes for vibe classification
│   ├── images/              # Downloaded product images
│   └── processed/           # Processed and cropped product images
├── models/
│   ├── catalog_faiss.index  # FAISS index for product matching
│   └── catalog_index.json   # Product ID mapping for the FAISS index
├── debug_outputs/           # Debug outputs for matched products
│   └── <video_name>/        # Subfolder for each processed video
├── debug_crops/             # Debug crops of detected objects
│   └── <video_name>/        # Subfolder for each processed video
├── scripts/
│   ├── classify_vibes.py    # Vibe classification using zero-shot learning
│   ├── match_products.py    # Product detection and matching
│   └── run_pipeline.py      # End-to-end pipeline
├── .gitignore
├── README.md
└── requirements.txt
```

## Configuration

The following parameters can be configured in `match_products.py`:

- `YOLO_BATCH_SIZE`: Number of frames per YOLO batch
- `CONF_THRESH`: Confidence threshold for detections (0-1)
- `IOU_THRESH`: IoU threshold for NMS (0-1)
- `FRAME_SKIP`: Process every N-th frame to speed up processing
- `TOP_K`: Number of nearest neighbors to return per query
- `SIMILARITY_EXACT`: Similarity threshold for exact matches
- `SIMILARITY_SIMILAR`: Similarity threshold for similar matches
- `SAVE_CROPS`: Whether to save debug crops of detections
- `CROPS_OUTPUT_DIR`: Directory to save debug crops
- `MATCHED_PRODUCTS_OUTPUT_DIR`: Directory to save matched product images

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Ultralytics YOLOv8
- OpenCV
- FAISS (for efficient similarity search)
- NumPy
- tqdm

## Debugging

The pipeline generates several debug outputs to help with troubleshooting:

1. **Debug Crops**: Saved in `debug_crops/<video_name>/` for each processed video
2. **Matched Products**: Saved in `debug_outputs/<video_name>/` for each video
3. **JSON Output**: Contains matched products and vibes in the specified output directory

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
