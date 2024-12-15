
<h1 align="center">
    Semantic segmentation </br> Applied on a high-resolution UAV street scene imagery
</h1>
<div style=width:100%>
    <img style="width:100%" src="assets/inf.gif" alt="gif of inference">
</div>

## Dataset

The dataset contains 270 paired samples, split into:

- **Training**: 200 samples
- **Validation**: 70 samples

Each paired sample consists of an image and its corresponding label or mask. Additionally, there are 150 single images provided for testing purposes.

The label images include annotations for 8 RGB classes, which are used for semantic segmentation. Each class is associated with a unique RGB value. 
The labels are later encoded as numbers ranging from 0 to 7.
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

### Aug_pre dataset

Data augmentation and preprocessing techniques were applied to generate additional data, making it ready for direct use in training.

- **Original resolution**: 540x960  
- **New resolution**: 256x512 
- **Total size**: 810 samples

This ***Aug_pre*** dataset is ready for immediate use in training.

<br>

### Download

|Resolution|Google Drive|Github|
|:--------:|:----------:|:----:|
|Orig(540 x 960)|[Download](https://drive.google.com/drive/folders/1qJzEsf-S0Kg2SSYELEBl4D9lo9ytIpax)|None
|Aug_pre(256 x 512)|None|[Download](https://github.com/pmohmmed/aug_pre/archive/refs/heads/main.zip)|

<br>

The Structure of dataset directory:
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

<br>

## Setup environment & load dataset
Create virtual environment (for more organization):


`python3 -m venv <path/to/env>`

<br>

Install python dependencies:

`pip install -r requirements.txt`

<br>

Clone aug_pre dataset:

`git clone https://github.com/pmohmmed/aug_pre.git`

<br>


## Augmentation
You can skip this stage if you are working with ***aug_pre*** dataset.

To proceed, run **augment.sh** script:
```bash 
bash scripts/augment.sh
```

-- OR --

Command:
```bash
python3 -m augment \
        --data_path dataset \
        --save_path aug_data \
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
python3 train.py \
        --data_path data_ip \
        --save_path ./model_w \
        --lr 0.0005 \
        --epochs 10 \
        --enc False \
        --res 256
```

<br>

## Inference

For this stage you will need:
- pre-trained model, ex: `unet.keras`, `inception.h5`
- preprocessed object, ex: `pre_obj.pkl`

Run **test.sh** script:
```bash
bash scripts/test.sh
```

-- OR --

Command:
```bash
python3 test.py \
        --data_path ../data_ip/ \
        --results_path results/\
        --model_path ./unet \
        --pre_path data/pre.pkl
```


