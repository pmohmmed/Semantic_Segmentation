import numpy as np

def select(images, labels, labels_to_check, classes_dic):
    """ Select the images & labels that meet the ratios on 'Classes_dic'"""

    selected_imgs = []
    selected_lbls = []
    number_of_classes = len(classes_dic)

    for i, lbl in enumerate(labels_to_check):
        ## number of pixels for each class
        total_pixels = lbl.size
        pixel_counts = {cls: np.sum(lbl == cls) for cls in classes_dic.keys()}

        ## count number of classes that meet their ratio in "classes_dic"
        count = sum(1 for cls, perc in classes_dic.items()
                    if (pixel_counts.get(cls, 0) / total_pixels) >= (perc))

        ## select only samples that meet all ratios specified in "classes_dic"
        if count == number_of_classes:
            selected_imgs.append(images[i])
            selected_lbls.append(labels[i])

    print(f"{len(selected_imgs)} samples selected")

    return selected_imgs, selected_lbls


def apply_aug(image, mask, aug_pip=None):
    if not aug_pip:
        return None, None

    augmented = aug_pip(image=image, mask=mask)
    return augmented["image"], augmented["mask"]


def augment(imgs, labels, aug_pip, n=5):
    """ augment each pair 'n' times """
    imgs_aug = []
    labels_aug = []

    for img, label in zip(imgs, labels):
      for _ in range (n):
        a_img, a_label = apply_aug(img, label, aug_pip)

        imgs_aug.append(a_img)
        labels_aug.append(a_label)

    return imgs_aug, labels_aug



