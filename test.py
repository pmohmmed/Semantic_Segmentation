from utils.options import test_opt
from utils.io import load, save

import numpy as np
import joblib
from tensorflow.keras.models import load_model


## command options
opt = test_opt()

## load data
test_dir = f'{opt.data_path}/test_data'
images, _ = load(test_dir, pre=False)
print(f'samples: {len(images)}')

## preprocess
pre = joblib.load(opt.pre_path)
pre_images = pre.images_pre(images)

## load model
print('\n--       Loading Model      --\n')
model = load_model(f'{opt.model_path}.keras')

## inference
print('\n--      Inference      --\n')
# predict
predicted_labels = model.predict(pre_images)

# select highist class for each pixel
encoded_labels = np.array([np.argmax(label, axis=-1)
                             for label in predicted_labels])

# decode back to rgb
labels = np.array([pre.decode_label(label) 
                     for label in encoded_labels], dtype='uint8')

## save results
save(opt.results_path, images, labels, prefix_name='inf_')

