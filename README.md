
<h1 align="center">
    Semantic segmentation | </br> Applied on a high-resolution UAV street scene imagery
</h1>
<div style=width:100%>
    <img style="width:100%" src="assets/inf.gif" alt="gif of inference">
</div>


## Dataset
### UAVid Dataset
The UAVid dataset is a high-resolution semantic segmentation dataset specifically designed for UAV-captured street scenes.

It consists of 42 sequences (seq1 to seq42) captured in 4K resolution with oblique views. Each sequence includes two folders: `Images` (input data) and `Labels` (ground truth).


<br>

### Dataset Distribution
The dataset is split into three subsets:

- **Train**: 200 samples
- **Validation**: 70 samples
- **Test**: 150 samples


<br>

### Label Information
The label images contain annotations for 8 semantic classes, each represented by a unique RGB value. For processing, these labels are encoded into numerical values ranging from 0 to 7.
|Class|R, G, B|Encode|
|-----|-------|:----:|
|Background clutter |![#000000](https://placehold.co/15x15/000000/000000.png) `(0, 0, 0)`     |`0`|   
|Building           |![#800000](https://placehold.co/15x15/800000/800000.png) `(128, 0, 0)`   |`1`|
|Road               |![#804080](https://placehold.co/15x15/804080/804080.png) `(128, 64, 128)`|`2`|
|Tree               |![#008000](https://placehold.co/15x15/008000/008000.png) `(0, 128, 0)`   |`3`|
|Low vegetation     |![#808000](https://placehold.co/15x15/808000/808000.png) `(128, 128, 0)` |`4`|
|Moving car         |![#400080](https://placehold.co/15x15/400080/400080.png) `(64, 0, 128)`  |`5`|
|Static car         |![#C000C0](https://placehold.co/15x15/C000C0/C000C0.png) `(192, 0, 192)` |`6`|
|human              |![#404000](https://placehold.co/15x15/404000/404000.png) `(64, 64, 0)`   |`7`|


<br>

### Variations
Two datasets were derived from the original UAVid dataset.

1. **uavid_512**:
    - Resolution: 512×1024.

2. **aug_enc_256**: 
    - Minimal preprocessing required.
    - Preprocessing steps:
        - **Augmentation**: Applied to the `train_data`.
        - **Label Encoding**: Applied to (train, val, and test) data.
    - New `train_data` size: 810 samples.
    - Resolution: 256×512.


<br>

### Download

|Dataset|Source|
|:---------|:----:|
|uavid (2160x3840)|[official](https://uavid.nl/)|
|aug_enc_256 (256x512)|[github](https://github.com/pmohmmed/aug_enc_256/archive/refs/heads/main.zip)|
|uavid_512 (512x1024)|[drive](https://drive.google.com/drive/folders/1NtKH4eGmYFExvNokf0Gs5FTt589VjEeY)|

<br>

The dataset is re-organized as follows:
```
dataset/
  ├── train_data/
  │   ├── Images/
  │   └── Labels/
  │
  ├── val_data/
  │   ├── Images/
  │   └── Labels/
  │
  └── test_data/
      └── Images/
```
**Note**: If you are working with the official UAVid dataset, use [this](https://github.com/pmohmmed/semantic-segmentation-uav/blob/main/notebooks/prepare_official_data.ipynb) notebook to re-organize it.

<br>

## Environment setup
- Clone this repo:
  ```bash
  git clone https://github.com/pmohmmed/semantic-segmentation-uavid.git;
  
  cd semantic-segmentation-uavid
  ```


  <br>

- Create and activate virtual environment (for more organization):

  ```bash
  python3 -m venv demo_env && source demo_env/bin/activate
  ```

<br>

- Install python dependencies:

  ```bash
  pip install -r requirements.txt
  ```

<br>

- Refer to the [Download](#download) section to download the dataset (if you haven’t already).

<br>

## Augmentation
Skip this stage if you are using the [aug_enc_256](#variations) dataset.

To proceed, run **augment.sh** script:
```bash 
bash scripts/augment.sh
```

-- OR --

Command:
```bash
python3 -m augment \
        --data_path path/to/dataset/train_data \
        --aug_data_path path/to/save/aug_data \
        --pre True \
        --res 256 \
```

<br>

## Train

Run **train.sh** script:
```bash
bash scripts/train.sh
```

-- OR --

Command:
```bash
## set --enc True if the data labels are RGB
python3 -m train \
        --lr 0.001 \
        --epochs 10 \
        --batch_size 16 \
        --enc False \
        --res 256 \
        --data_path path/to/dataset/ \
        --model_path path/to/save/model/unet.keras \
```

<br>

## Inference

Run **test.sh** script:
```bash
bash scripts/test.sh
```

-- OR --

Command:
```bash
python3 -m test \
        --data_path path/to/test_data \
        --results_path path/to/save/results/\
        --model_path path/to/model/unet.keras \
        --show_results False
``` 
<br>

## Upcoming Work
- Evaluate model performance using metrics such as IoU, SSIM, mIoU, F1-Score, etc.
- Experiment with different architectures to improve accuracy and efficiency.
- Expand and refine the dataset for better generalization.
- Implement and optimize real-time inference capabilities.

<br>

## Licence
The dataset and all variations are licensed under the [UAVid](https://uavid.nl/) dataset license:

`Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License`

[You may not use this work for commercial purposes]
