# DeShadowing
## Overview
This project addresses the challenge of shadow removal in images using traditional methods, improving upon the limitations of previous approaches that often failed to achieve satisfactory performance. By leveraging classical image processing techniques such as segmentation, clustering, and histogram matching, this pipeline demonstrates robust shadow detection and removal without relying on trained models. We introduce three features gradient, texture, and center distance to classify shadowed and unshadowed regions and restore shadowed areas using paired unshadowed regions. This method highlights the potential of traditional techniques to overcome past shortcomings, offering a practical solution without the need for extensive training datasets.
## Requirements
## Experiment Results
## Structure
```
.
├── data/                    # Example input images
│   └── shadow               # Example shadow images
│   └── shadow_free          # Shadow remove ground truth
├── main.py                  # Main apprroch include shadow detect and removal
├── output/                  # Intermediate output images
│   └── Detect               # Results of detect images
│   └── Remove               # Results of remove images
│   └── Paired               # Results of paired regions images
```
