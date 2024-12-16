from utils.helper import message
from models.unet import build_unet
from utils.options import train_opt
from utils.io import load
from data.preprocessor import Preprocessor

import joblib
from keras.optimizers import Adam


## command options
opt = train_opt()
a_pre = not(opt.enc) # already preprocssed

## load data
train_dir = f'{opt.data_path}/train_data'
val_dir = f'{opt.data_path}/val_data'

images, labels = load(train_dir, pre=a_pre, shuffle=True, name='Train data')
images_val, labels_val = load(val_dir, pre=a_pre, name='Val data')
print(f'train: {len(images)}')
print(f'val: {len(images_val)}')

## preprocess
pre = Preprocessor(resize_to=(opt.res, 2*opt.res), one_h=False)
joblib.dump(pre, opt.pre_path)

# train
pre_images = pre.images_pre(images)
pre_labels = pre.labels_pre(labels, fix=opt.enc ,encode=opt.enc)
print(f'train: {pre_images.shape}, {pre_labels.shape}')

# val
pre_images_val = pre.images_pre(images_val)
pre_labels_val = pre.labels_pre(labels_val, fix=opt.enc ,encode=opt.enc)
print(f'val: {pre_images_val.shape}, {pre_labels_val.shape}')

## build model
input_shape = pre_images.shape[1:]
num_classes = 8
model = build_unet(input_shape, num_classes)

## compile
model.compile(
    optimizer=Adam(learning_rate=opt.lr),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

if opt.mods:
    model.summary()


## train
message('Training...')
u_history = model.fit(
        pre_images[:10], pre_labels[:10],
        validation_data=(pre_images_val[:10], pre_labels_val[:10]),
        epochs=opt.epochs,
        )

## save
#model.save_weights(f'{opt.model_path}.weights.h5')
model.save(opt.model_path)
