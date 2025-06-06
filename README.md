# Fashion Finder

A computer vision pipeline for detecting and categorizing fashion items in product images.

## Project Overview

This project implements a pipeline for processing fashion product images, detecting clothing items, and categorizing them. It's designed to work with e-commerce product catalogs and can be used for tasks like visual search, recommendation systems, and inventory management.

## Features

- **Image Download**: Download product images from a catalog
- **Object Detection**: Detect fashion items using YOLOv8
- **Category Filtering**: Filter detections by clothing categories
- **Batch Processing**: Process multiple products with configurable limits

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

### 1. Download Product Images

```bash
python scripts/download_images.py --limit 10
```

### 2. Detect and Crop Fashion Items

```bash
# Process all images with default settings (clothing only)
python scripts/detect_objects.py --input data/images --output data/processed

# Process with custom settings
python scripts/detect_objects.py \
    --input data/images \
    --output data/processed \
    --limit 5 \
    --classes clothing bags
```

### 3. Process Catalog Data (Optional)

```bash
python scripts/data_ingest.py
```

## Project Structure

```
├── data/                    # Data directory (not versioned)
│   ├── images/             # Downloaded product images
│   └── processed/          # Processed and cropped images
├── scripts/                 # Python scripts
│   ├── data_ingest.py      # Data processing utilities
│   ├── detect_objects.py   # Object detection pipeline
│   └── download_images.py  # Image downloader
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Pandas
- Requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
