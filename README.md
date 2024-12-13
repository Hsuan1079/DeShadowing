## Overview
This project addresses the challenge of shadow removal in images using traditional methods, improving upon the limitations of previous approaches that often failed to achieve satisfactory performance. By leveraging classical image processing techniques such as segmentation, clustering, and histogram matching, this pipeline demonstrates robust shadow detection and removal without relying on trained models. We introduce three features gradient, texture, and center distance to classify shadowed and unshadowed regions and restore shadowed areas using paired unshadowed regions. This method highlights the potential of traditional techniques to overcome past shortcomings, offering a practical solution without the need for extensive training datasets.
## Requirements

```
python 3.9.16
```
1. Clone the repository
   ```
   git clone https://github.com/Hsuan1079/DeShadowing.git
   ```
2. Run the main code (you can change the image to your own)
   ```
   python main.py
   ```
## Methodology
### Shadow Detect
![image](https://github.com/user-attachments/assets/a15d76dc-539b-4f2d-8079-5923d075b32f)

### Shadow Removal
<img width="1192" alt="image" src="https://github.com/user-attachments/assets/e30e3744-00a9-4ba2-8076-277bb54a6b4a" />


## Experiment Results
| Original Image | Remove Image | Ground Truth |
|----------------|--------------|--------------|
| ![Original Image](https://github.com/user-attachments/assets/962187d0-e399-48ed-bf28-5aaa1f3f0503) | ![Remove Image](https://github.com/user-attachments/assets/50c9d4d4-7de1-4d18-9cb1-63a6a4bfcc7b) | ![Ground Truth](https://github.com/user-attachments/assets/31aca6f4-98e1-410e-a7a4-b201c51236ea) |
| ![Original Image](https://github.com/user-attachments/assets/5afea2d1-1b14-42fd-8609-7c8c68d3039e) | ![Remove Image](https://github.com/user-attachments/assets/22c8fbab-3a3b-4625-8352-a21def9b15d4)| ![Ground Truth](https://github.com/user-attachments/assets/aa7dc5ed-2997-47a4-b739-5f0e0087f09a) |

## Metric Result
<img width="1187" alt="image" src="https://github.com/user-attachments/assets/372f71e6-46ad-46c3-a82e-a0b9fc076283" />

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
