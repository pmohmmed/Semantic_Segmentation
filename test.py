from utils.helper import stg_msg, save_msg
from utils.options import test_opt
from utils.io import load, save
from utils.inference import infer
from data.preprocessor import Preprocessor

import numpy as np
#import joblib
from tensorflow.keras.models import load_model



## vars
opt = test_opt()

test_dir = opt.data_path

## load
stg_msg('Loading')

# data
images, _ = load(test_dir, pre=False, name='Test data')
print(f'# loaded imgs: {len(images)}')

# model 
model = load_model(opt.model_path)
print(f'# model: {model.name}')

# preprocessor object
#pre = joblib.load(opt.pre_obj_path)
pre = Preprocessor(resize_to=model.input_shape[1:-1], one_h=False)


## preprocess
stg_msg('Preprocessing')
pre_images = pre.images_pre(images)

print('(inference shape)')
print(f'# imgs: {pre_images.shape}')


## inference
stg_msg('Inferencing', c='.')
encoded_labels = infer(pre_images, model)


## save results
if len(encoded_labels):
    # decode back to rgb
    labels = np.array([pre.decode_label(label) 
                       for label in encoded_labels], dtype='uint8')

    # save rgb labels
    stg_msg('Saving')
    save(opt.results_path, pre_images*255, labels, prefix_name='inf_')
    save_msg(f'inference results saved to: {opt.results_path}')

