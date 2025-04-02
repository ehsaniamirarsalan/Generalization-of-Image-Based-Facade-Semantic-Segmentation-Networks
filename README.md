# :cityscape: Generalization-of-Image-Based-Facade-Semantic-Segmentation-Networks
This repository contains the implementation and evaluation of three neural network architectures for facade semantic segmentation: Mask R-CNN, DeepLabV3+, and U-Net. The primary goal is to assess their generalization capabilities across different geographic regions, specifically focusing on how models trained on European facade datasets perform when applied to distinctly different architectural styles.
## Project Overview

Urban environments vary significantly in architectural characteristics across different regions. This research systematically evaluates how networks trained on European facade datasets perform when applied to buildings in Munich, Germany, and Singapore. The study aims to:

1. Evaluate cross-regional generalization capabilities
2. Compare model robustness across geographic domain shifts
3. Identify key factors affecting generalization
4. Provide strategies to enhance cross-domain generalization

## Models Implemented :computer:

### Mask R-CNN
- Instance segmentation approach that detects each facade element as a separate instance
- ResNet-50 FPN backbone initialized with ImageNet weights
- Strong performance on discrete elements like windows, doors, and decorations

### DeepLabV3+
- Semantic segmentation model with ASPP module for multi-scale context
- ResNet-50 backbone with atrous convolutions
- Performs well on large continuous regions like facades/walls

### U-Net
- Encoder-decoder architecture with skip connections
- Lightweight model trained from scratch
- Effective baseline for facade segmentation tasks

## Datasets :bookmark_tabs:

### Training Datasets
- **CMP Facade Dataset**: 606 rectified images of building facades from European cities with 12 semantic classes
- **eTRIMS Dataset**: 60 images of building facades with 8 semantic classes

### Test Datasets
- **TUM2TWIN**: Images from Munich, Germany (Technical University of Munich)
- **Singapore Urban Building Facade Dataset**: Images from Temple Street, Singapore

## Key Findings :key:

- Mask R-CNN outperformed other models in 9 out of 11 facade classes by IoU metrics
- DeepLabV3+ showed strong performance in facade/wall classification
- U-Net performed best on door class segmentation
- All models exhibited poor cross-regional generalization when applied to Singaporean facades
- Architectural style variations and dataset biases significantly impact model performance

## Evaluation Metrics :bar_chart:

The models are evaluated using:
- Intersection over Union (IoU)
- Mean IoU (mIoU)
- Pixel Accuracy
- F1 Score
- Confusion Matrix

## Repository Structure :notebook:

```
├── Mask RCNN/           # Mask R-CNN implementation
│   └── Read.md          # Link to Google Colab implementation
├── DeepLabV3+/          # DeepLabV3+ implementation
│   └── Read.md          # Link to Google Colab implementation
├── U-Net/               # U-Net implementation
│   └── Read.md          # Link to Google Colab implementation
├── Data/                # Dataset information
│   └── read.md          # Notes on CMP facade dataset
├── dashbord.py          # Visualization dashboard for result analysis
├── evaluation.py        # Metrics calculation and evaluation scripts
└── README.md            # Project documentation
```

## Usage

1. **Access the model implementations** via the Google Colab links provided in the respective directories.
2. **Prepare the datasets**:
   - Download the CMP facade dataset
   - Store images and masks in your Google Drive
   - Adjust the paths within the code to point to your data
3. **Train the models** using the provided notebooks
4. **Evaluate performance** using the evaluation.py script
5. **Visualize results** with the dashboard.py script

## Results Dashboard

- Overall network performance comparison
- Class-wise IoU and F1 metrics
- Radar charts for model comparison
- Performance line charts

## Acknowledgments

This project is part of the research conducted at the Technical University of Munich, Professorship of Photogrammetry and Remote Sensing.
- Advisor: Olaf Wysocki, Dr.-Ing.
