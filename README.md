# Fish Multi-Object Tracking Pipeline

A comprehensive deep learning pipeline for multi-object tracking of fish in the GMOT40 dataset using Faster R-CNN object detection combined with IoU/appearance-based Hungarian tracking.

## Overview

This project consists of two main Jupyter notebooks:

1. **Segmentation_part.ipynb** - Training pipeline for the segmentation encoder and detector
2. **Tracking_part.ipynb** - Full evaluation pipeline with GMOT40 benchmark comparison

## Key Features

- **Faster R-CNN Detector**: Trained directly on GMOT40 ground-truth boxes for individual fish detection
- **Appearance Embedder**: Leverages fine-tuned VAE-SA-UNet ConvNeXt-Small encoder for appearance features
- **Hybrid Tracker**: Combines IoU-based spatial matching (70%) with appearance similarity (30%) using Hungarian algorithm
- **Held-out Evaluation**: Chronological per-sequence train/test split ensures genuine out-of-sample evaluation
- **CLEAR-MOT Metrics**: Full motmetrics integration for comprehensive tracking benchmark

## Performance

### Held-out Test Results (Last 20% of Frames, Never Seen by Detector)

| Metric | Value |
|--------|-------|
| **MOTA** | 81.9% |
| **IDF1** | 71.5% |
| **GT Objects** | 171 |
| **ID Switches** | ~50 |
| **Fragmentations** | ~34 |

**Paper Comparison (GMOT40 Fish Category, Table 5):**
- Outperforms baseline IoU/VIoU trackers (76-77% MOTA)
- Competitive with DeepSort (81.2% MOTA)
- Significantly improves over watershed-based segmentation approach (−72.8% MOTA)

## Dataset Structure

```
GMOT40/
├── GenericMOT_JPEG_Sequence/     # Frame sequences
│   ├── fish-1/img1/*.jpg
│   ├── fish-2/img1/*.jpg
│   ├── fish-3/img1/*.jpg
│   └── fish-4/img1/*.jpg
└── track_label/                   # Ground truth annotations (MOTChallenge format)
    ├── fish-1.txt
    ├── fish-2.txt
    ├── fish-3.txt
    └── fish-4.txt
```

## Installation

### Requirements

```
torch
torchvision
timm
albumentations
opencv-python
scipy
pandas
matplotlib
motmetrics
tqdm
```

### Setup

```bash
# Install PyTorch (CUDA 11.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install timm albumentations opencv-python scipy pandas matplotlib motmetrics tqdm
```

### GMOT40 Dataset

1. Download GMOT40 dataset from the official source
2. Place the extracted `GMOT40/` folder in one of these locations:
   - `/teamspace/studios/this_studio/`
   - `/kaggle/input/`
   - `/content/`
   - Home directory
   - Notebook working directory
3. Alternatively, upload `GMOT40.zip` and the notebook will auto-extract it

## Usage

### Notebook 1: Training Pipeline

`fish_image_pipeline_with_tracking_50epochs(95.59).ipynb`

**Steps:**
1. Setup and configuration
2. Load GMOT40 dataset
3. Define segmentation encoder (VAE-SA-UNet with ConvNeXt-Small)
4. Train Faster R-CNN detector on train subset
5. Evaluate on full sequences
6. Generate qualitative visualizations

**Output:**
- Trained detector checkpoint
- Segmentation encoder checkpoint (pre-trained)
- Training curves and metrics

### Notebook 2: Evaluation & Benchmark

`fish_pipeline_gmot40_table5_comparison(run).ipynb`

**Steps:**
1. Load pre-trained components (detector, encoder, embedder)
2. Apply chronological held-out split per sequence
3. Run detection + tracking on full sequences
4. Score only held-out test tail (never seen by detector)
5. Compare against paper benchmarks (Table 5)
6. Generate publication-quality visualizations

**Output:**
- CLEAR-MOT metrics (MOTA, IDF1, MT, ML, etc.)
- Table 5 comparison CSV
- Qualitative tracking visualizations (frames + 3D trajectories)

## Architecture Details

### Segmentation Encoder (Appearance Extraction)

```
VAE-SA-UNet
├── ConvNeXt-Small Encoder (pretrained)
├── VAE Bottleneck (mu, logvar)
├── Self-Attention Bridge
│   ├── Learned Positional Encoding
│   └── Multi-head Self-Attention Layers
└── Decoder + Attention Gates
```

### Faster R-CNN Detector

- **Backbone:** ResNet-50 + FPN
- **Classes:** 2 (background + fish)
- **Input Size:** Variable (320×320 for training crops)
- **Confidence Threshold:** 0.5

### FishTracker

**Hybrid Score:** `score = 0.70 × IoU + 0.30 × cosine_similarity`

**Parameters:**
- `min_score`: 0.3
- `max_age`: 3 frames
- `min_hits`: 1 (emit track immediately)
- `ema_alpha`: 0.9 (appearance embedding smoothing)

## Key Fixes (from Original)

1. **Held-out Evaluation Fix**: Original split was frame-level random (80/20), allowing detector to train on 80% of test frames. Now uses chronological per-sequence split — first 80% of each sequence for training, last 20% exclusively for evaluation.

2. **Detection Architecture**: Replaced segmentation + watershed approach (which cannot separate touching fish in dense schools) with Faster R-CNN for direct box detection.

3. **Tracker Hyperparameters**: Optimized IoU/appearance fusion weights (70/30 split) via hyperparameter sweep to balance IDF1, MOTA, and ID switch count.

## Results Visualization

### Qualitative Examples

- **Figure 5 Style:** Side-by-side tracked frames with per-ID color coding
- **3D Trajectories:** Ground truth vs. predicted spatial-temporal tracks

### Metrics

- Full CLEAR-MOT breakdown per sequence
- Comparison with watershed baseline and DeepSort
- ID switch and fragmentation analysis

## Checkpoints

By default, trained models are saved to `./checkpoints/`:

```
checkpoints/
├── best_vae_sa_unet.pth          # Segmentation encoder
├── best_detector.pth             # Faster R-CNN detector
├── gmot40_heldout_benchmark_final.csv
└── table5_comparison.csv
```

## Citation

If using this pipeline in research, cite:

```
Faster R-CNN detector + IoU/appearance Hungarian tracker
Evaluated on GMOT40 (fish category) with held-out chronological split
MOTA: 81.9%, IDF1: 71.5% (on held-out test frames)
```

## References

- **GMOT40 Dataset**: [Generic Multi-Object Tracking Dataset](https://github.com/Zhongdao/GMOT-Tracking)
- **Faster R-CNN**: [Ren et al., 2016](https://arxiv.org/abs/1506.01497)
- **CLEAR-MOT**: [Bernardin & Stiefelhagen, 2008](https://ieeexplore.ieee.org/document/4756141)
- **motmetrics**: [Tracking Metrics Library](https://github.com/cheind/py-motmetrics)

## License

Academic use. Please refer to dataset and library licenses.

## Troubleshooting

**GMOT40 not found:**
- Ensure dataset is in one of the search paths
- Check file structure matches expected layout

**Out of memory errors:**
- Reduce `BATCH_SIZE` in Config (default: 24)
- Enable gradient checkpointing if available
- Use smaller image sizes

**Low tracking performance:**
- Verify detector accuracy first (check detection crops)
- Adjust `min_score` threshold (lower = more detections)
- Tune `iou_weight` / `appearance_weight` balance

---

**Last Updated:** July 2026  
**Status:** Complete — ready for publication
