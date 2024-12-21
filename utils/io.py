from utils.helper import stg_msg

import numpy as np
from glob import glob
import cv2
import os


def prepare_paths(folder_path=None, pre=False):

    images_dir = f"{folder_path}/Images/"
    labels_dir = f"{folder_path}/Labels/"

    image_paths = sorted(glob(os.path.join(images_dir, "*.png")))
    label_paths = sorted(glob(os.path.join(labels_dir, "*.png")))

    return image_paths, label_paths


def load_img(path):

    if path.find('Labels/pre')>-1:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)


def load(folder_path=None, shuffle=False, pre=False, name='Data'):
    #stg_msg(f'Loading {name}')

    image_paths, label_paths = prepare_paths(folder_path, pre)

    images = [load_img(img_path) for img_path in image_paths]
    labels = [load_img(lbl_path) for lbl_path in label_paths]

    if shuffle:
        images, labels = shuffle_data(images, labels)

    #print('--> Compeleted')
    if pre:
        return np.array(images), np.array(labels)
    return images, labels


def load_seq(path):
    sequences = glob(os.path.join(path, 'seq*'))
    images, labels = [], []
    for seq in sequences:
        imgs, lbls = load(seq, pre=False)
        
        images += imgs
        labels += lbls

    return images, labels


def shuffle_data(images, labels):
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    images_shuff = [images[i] for i in indices]
    labels_shuff = [labels[i] for i in indices]

    return images_shuff, labels_shuff


def save(path, images, labels, prefix_name='__'):
    """Prefix_name should be "pre_" if you're saving preprocessed data"""
    
    #stg_msg('Saving')

    images_dir = f"{path}/Images"
    labels_dir = f"{path}/Labels"

    os.makedirs(images_dir, exist_ok=True)

    if labels is not None:
        os.makedirs(labels_dir, exist_ok=True)

    for idx, image in enumerate(images):
        ## image
        image = image.astype('uint8')
        image_path = os.path.join(images_dir, f"{prefix_name}{idx}.png")

        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(image_path, image)
            

        ## label
        if labels is not None:
            label = labels[idx].astype('uint8')
            label_path = os.path.join(labels_dir, f"{prefix_name}{idx}.png")

            if label.ndim == 3 and label.shape[-1] == 3:
                label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)

            cv2.imwrite(label_path, label)

