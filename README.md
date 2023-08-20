# SR4ZCT

This repository contains the code accompanying the paper:

**SR4ZCT: Self-supervised Through-plane Resolution Enhancement for CT Images with Arbitrary Resolution and Overlap**

## Abstract
Computed tomography (CT) is a widely used non-invasive medical imaging technique for disease diagnosis. The diagnostic accuracy is often affected by image resolution, which can be insufficient in practice. For medical CT images, the through-plane resolution is often worse than the in-plane resolution and there can be overlap between slices, causing difficulties in diagnoses. Self-supervised methods for through-plane resolution enhancement, which train on in-plane images and infer on through-plane images, have shown promise for both CT and MRI imaging. However, existing self-supervised methods either neglect overlap or can only handle specific cases with fixed combinations of resolution and overlap. To address these limitations, we propose a self-supervised method called SR4ZCT. It employs the same off-axis training approach while being capable of handling arbitrary combinations of resolution and overlap. Our method explicitly models the relationship between resolutions and voxel spacings of different planes to accurately simulate training images that match the original through-plane images. We highlight the significance of accurate modeling in self-supervised off-axis training and demonstrate the effectiveness of SR4ZCT using a real-world dataset.

## Prerequisites

- Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/SR4ZCT.git
   cd SR4ZCT
   ```

2. Create and activate a Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate sr4zct
    ```

## How to Use
This code includes example data from the [Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/#trainingData). However, you're free to use your own data.

### Steps
1. **Data Preparation**: Generate training data based on the original resolution/overlap and target resolution/overlap. Also, prepare the test data.
    ```python
    python 0_prepare_volume.py
    ```

2. **Training and Testing**: Run the training and testing process. Intermediate results will be saved in the predefined directory *L291_result/* for monitoring the training progress. The enhanced resolution volumes will also be stored here.

## Results


## Citation
If you find this work useful and use it in your research, please consider citing our paper:
```
```