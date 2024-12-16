from utils.helper import stg_msg, save_msg
from utils.io import load, save
from utils.aug import select, augment
from data.preprocessor import Preprocessor
from utils.options import aug_opt

import albumentations as A
import numpy as np
import joblib



## vars
opt = aug_opt() # arguments passed to the command

pre = Preprocessor(resize_to=None, one_h=False)
pre_aug = Preprocessor(resize_to=(opt.res, opt.res*2), one_h=False)


## load
stg_msg('Loading')

images, labels = load(f'{opt.data_path}/train_data', pre=False)
print(f'# train data: {len(images)}')


## encode labels
stg_msg('Prepare labels')

if len(labels) and labels[0].ndim == 3: 
    pre_labels = pre.labels_pre(labels, fix=True, encode=True)
else:
    pre_labels = np.array(labels)
    labels = [pre.decode_label(lbl) for lbl in pre_labels]


## select samples to be augmented
stg_msg('Selecting sampels')

# criterias
minor_lbls = { 
    5:0.05,  # Moving car
    6:0.02, # Static car
    7:0.03  # Human
  }

# select
images_sel, labels_sel = select(images, labels, pre_labels, minor_lbls)
print(f'# selected samples: {len(images_sel)}')


## augmentation
stg_msg('Augmentation', c='.')

# number of augmentations per sample
while True:
    try:
        n = int(input("$ How many variations per sample?\n: "))
        break  
    except ValueError:
        print("Invalid input! please enter a valid integer.\n")## preprocess


# transformations     
aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomCrop(height=256, width=256, p=0.4),
    A.ElasticTransform(p=0.2),

    A.OneOf([
        A.RandomBrightnessContrast(p=0.7),
        A.ColorJitter(p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ], p=0.7),

], additional_targets={"mask": "mask"})

# augment
images_aug, labels_aug = augment(images_sel, labels_sel, aug, n)
print(f'# generated data: {len(images_aug)}')

# add to original samples
total = len(images_aug) + len(images)
q = input(f"\n$ Extend with original data (total size will be '{total}')?\n: ")

accept = ['yes', 'y', 'add', 'ok', '1', 'yup']
if q.lower() in accept:
    images_aug += images
    labels_aug += labels


## preprocess before save (option)
stg_msg('Preprocessing (optionally)')

prefix = 'aug_'
if opt.pre:
    images_aug = pre_aug.resize(images_aug)
    labels_aug = pre_aug.labels_pre(labels_aug, fix=True, encode=True)

    print('(train shape)')
    print(f'# imgs: {images_aug.shape}\n# lbls {labels_aug.shape}')
    prefix = 'pre_'


## save
stg_msg('Saving')

# augmented data
save(opt.aug_data_path, images_aug, labels_aug, prefix_name=prefix)
save_msg(f'augmented data saved to: {opt.aug_data_path}')

# preprocessor object
joblib.dump(pre_aug, opt.pre_obj_path)
save_msg(f'preprocessor object saved to: {opt.pre_obj_path}')

