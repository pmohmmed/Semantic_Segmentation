import numpy as np
import cv2


class_labels = {
    (0, 0, 0): 0,          # Background clutter
    (128, 0, 0): 1,        # Building
    (128, 64, 128): 2,     # Road
    (0, 128, 0): 3,        # Tree
    (128, 128, 0): 4,      # Low vegetation
    (64, 0, 128): 5,       # Moving car
    (192, 0, 192): 6,      # Static car
    (64, 64, 0): 7         # Human
}


class Preprocessor:
    def __init__(self, resize_to:tuple, one_h:bool):
      self.target_size = resize_to
      self.one_hot = one_h


    def encode_label(self, rgb_label:np.ndarray):
        """
        Map each pixel of the label a class label
        """
        encoded_label = np.zeros(rgb_label.shape[:2], dtype=np.uint8)

        ## replace each rgb pixel with class label
        for rgb, label in class_labels.items():
            mask = (rgb_label == np.array(rgb)).all(axis=-1)  # boolean mask
            encoded_label[mask] = label  # encode

        return encoded_label


    def encode_labels(self, labels:np.ndarray):
        return [self.encode_label(label) for label in labels]


    def apply_one_hot(self, labels):
      return one_hot(labels, depth=8, axis=-1).numpy()


    def decode_label(self, encoded_label):

        rgb_label = np.zeros((*encoded_label.shape, 3), dtype=np.uint8)

        for rgb, label in class_labels.items():
            mask = encoded_label == label # boolean mask 
            rgb_label[mask] = np.array(rgb).astype(np.uint8)

        return rgb_label


    def nearest_rgb(self, rgb_pixel):
        colors = np.array(list(class_labels.keys()))
        distances = np.linalg.norm(colors - rgb_pixel, axis=1) 
        nearest_color = colors[np.argmin(distances)]

        return tuple(nearest_color)


    def fix_label(self, rgb_label:np.ndarray):
        """ Map pixels the isn't one of the 8 classes to the closest pixel """
        input_label = rgb_label.copy()

        ## array of lists(rgb)
        valid_colors = np.array(list(class_labels.keys()))

        ## flat all the pixels
        flat = input_label.reshape(-1, 3) # shape: (H * W, 3)

        ## pixels that aren't of the 8 classes
        invalid_mask = ~np.any((flat[:, None] == valid_colors).all(axis=-1),
                               axis=1)

        ## find and replace the invalid pixels with nearest pixels
        if np.any(invalid_mask):
            invalid_pixels = flat[invalid_mask]

            nearest_colors = np.array([self.nearest_rgb(pixel) for pixel in invalid_pixels])

            flat[invalid_mask] = nearest_colors

        ## original shape
        return flat.reshape(rgb_label.shape)


    def fix_labels(self, labels:np.ndarray):
        return [self.fix_label(label) for label in labels]



    def resize(self, imgs, target=None):

        if target is None:
            if self.target_size is None:
                return imgs
            else:
                target = self.target_size

        if type(target) is not tuple:
            target = target, target
        else:
            target = target[1], target[0]

        return np.array([cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
                         for img in imgs], dtype='uint8')


    def images_pre(self, images):
        print('\n--       Images Preprocessing       --\n')
        
        ## resize & normalize
        images = self.resize(images) / 255.0 

        return np.array(images)


    def labels_pre(self, labels, encode=True, fix=True):
        print('\n--       Labels Preprocessing      --\n')

        ## resize
        pre_labels = self.resize(labels)

        ## fix pixels (if there's pixels out of the 8 classes)
        if fix:
          pre_labels = self.fix_labels(pre_labels)

        ## encoding
        if encode:
          pre_labels = self.encode_labels(pre_labels) # 0, 1, 2, ...

        ## on_hot encoding
        if self.one_hot == True:
          pre_labels = self.apply_one_hot(pre_labels)

        if self.target_size is None:
          return pre_labels

        return np.array(pre_labels)

