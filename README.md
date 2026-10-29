# AI Identity Transformation Studio

A modular computer vision and generative AI studio for face swapping, identity synthesis, and full-body character transformation.

## Project Structure
```
video_swap/
├── pyproject.toml              # Build config and pytest path settings
├── requirements.txt            # Core dependencies
├── src/
│   └── identity_studio/
│       ├── core/               # Image loading, color conversions, geometry, timer
│       ├── detectors/          # Pluggable face detectors (Haar, YuNet DNN)
│       └── pipeline/           # Detection annotator and benchmarking runner
├── data/
│   ├── inputs/                 # Curated test images (with and without faces)
│   └── outputs/                # Annotated output images
├── docs/                       # Benchmark and verification reports
├── models/
│   └── weights/                # Pretrained detector weights (YuNet ONNX)
├── scripts/
│   └── run_phase1.py           # CLI benchmark runner
└── tests/                      # Automated test suite
```

## Quickstart

### 1. Environment Setup
```bash
# Create virtual environment with Python 3.12
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Phase 1 Face Detection & Latency Benchmark
```bash
# Using OpenCV Haar Cascade detector:
python scripts/run_phase1.py --detector haar

# Using OpenCV DNN YuNet model:
python scripts/run_phase1.py --detector yunet
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```
