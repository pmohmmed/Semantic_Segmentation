from utils.helper import stg_msg, save_msg
from models.unet import build_unet
from utils.options import train_opt
from utils.io import load
from data.preprocessor import Preprocessor

import joblib
from keras.optimizers import Adam
from os.path import join



## vars
opt = train_opt() # command arguments

train_dir = join(opt.data_path, 'train_data')
val_dir = join(opt.data_path, 'val_data')

pre = Preprocessor(resize_to=(opt.res, 2*opt.res), one_h=False)
a_pre = not(opt.enc) # already preprocssed


## load
stg_msg('Loading')

# train
images, labels = load(train_dir, pre=a_pre, shuffle=True, name='Train data')
print(f'# train data: {len(images)}')

# val
images_val, labels_val = load(val_dir, pre=a_pre, name='Val data')
print(f'# val data: {len(images_val)}')


## preprocess
stg_msg('Preprocessing')

# train
pre_images = pre.images_pre(images)
pre_labels = pre.labels_pre(labels, fix=opt.enc ,encode=opt.enc)

print('(train shape)')
print(f'# imgs: {pre_images.shape}\n# lbls:{pre_labels.shape}')


# val
pre_images_val = pre.images_pre(images_val)
pre_labels_val = pre.labels_pre(labels_val, fix=opt.enc ,encode=opt.enc)

print('\n(val shape)')
print(f'# imgs: {pre_images_val.shape}\n# lbls {pre_labels_val.shape}')


## build and compile the model
stg_msg('Build & Compile')

# build architecture
input_shape = pre_images.shape[1:]
num_classes = 8
model = build_unet(input_shape, num_classes) # model

# compile
model.compile(
    optimizer=Adam(learning_rate=opt.lr),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
if opt.mods:
    model.summary()


## train
stg_msg('Training', c='.')

u_history = model.fit(
        pre_images, pre_labels,
        validation_data=(pre_images_val, pre_labels_val),
        batch_size=opt.batch_size,
        epochs=opt.epochs,
        )


## save
stg_msg('Saving')

# preprocessor object
joblib.dump(pre, opt.pre_obj_path)
save_msg(f'preprocessor object saved to: {opt.pre_obj_path}')

# model
model.save(opt.model_path)
save_msg(f'model saved to: {opt.model_path}')
