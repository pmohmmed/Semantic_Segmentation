from utils.io import load, save
from utils.aug import select, augment
from data.preprocessor import Preprocessor
from utils.options import aug_opt

import albumentations as A


## command options
opt = aug_opt()

## load
images, labels = load(f'{opt.data_path}/train_data', pre=False)
print(f'samples: {len(images)}')

## encode labels
pre = Preprocessor(resize_to=None, one_h=False)
pre_labels = pre.labels_pre(labels, fix=True, encode=True)

## select samples to be augmented
# criterias
minor_lbls = { 
    5:0.05,  # Moving car
    6:0.02, # Static car
    7:0.03  # Human
  }

# select
images_sel, labels_sel = select(images, labels, pre_labels, minor_lbls)

## number of augmentations per sample
while True:
    try:
        n = int(input("\nHow many variations per sample you want to apply:\n"))
        break  
    except ValueError:
        print("Invalid input! Please enter a valid integer.")## preprocess


## transformations     
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

images_aug, labels_aug = augment(images_sel, labels_sel, aug, n)
print(f'aug imgs: {len(images_aug)}')


## add to original samples
total = len(images_aug) + len(images)
q = input(f"Add to original data ?, total size will be {total}\n: ").lower()

accept = ['yes', 'y', 'add', 'ok', '1', 'yup']
if q in accept:
    images_aug += images
    labels_aug += labels


## preprocess before save (option)
prefix = 'aug_'
if opt.pre:
    pre = Preprocessor(resize_to=opt.res, one_h=False)
    images_aug = pre.resize(images_aug)
    labels_aug = pre.labels_pre(labels_aug, fix=True, encode=True)
    print(f'imgs: {images_aug.shape}')
    print(f'lbl: {labels_aug.shape}')
    prefix = 'pre_'


## save
save(opt.save_path, images_aug, labels_aug, prefix_name=prefix)
