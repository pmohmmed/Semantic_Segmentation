from tensorflow.keras import layers
from tensorflow.keras.models import Model


def conv(f_map, num_filters, k, padd, a):
    x = layers.Conv2D(num_filters, k, padding=padd)(f_map)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(a)(x)
    return x

def conv_block(f_map, num_filters, k=3, padd='same', a='relu', num_conv=2):
    for _ in range(num_conv):
        f_map = conv(f_map, num_filters, k, padd, a)
    return f_map

def encoder_block(f_map, num_filters, dropout_rate=0.3):
    x = conv_block(f_map, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    if dropout_rate:
        p = layers.Dropout(dropout_rate)(p)
    return p, x

def decoder_block(f_map, skip_con, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(f_map)
    x = layers.Concatenate()([x, skip_con])
    x = conv_block(x, num_filters)
    return x

def build_unet(inputs_shape, num_classes):
    inputs = layers.Input(inputs_shape)

    ## encoding
    x1, s1 = encoder_block(inputs, 64)
    x2, s2 = encoder_block(x1, 64)
    x3, s3 = encoder_block(x2, 128)
    x4, s4 = encoder_block(x3, 256)

    ## bottleneck
    b = conv_block(x4, 512)

    ## decoding
    x4 = decoder_block(b, s4, 256)
    x3 = decoder_block(x4, s3, 128)
    x2 = decoder_block(x3, s2, 64)
    x1 = decoder_block(x2, s1, 64)

    ## output
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x1)

    return Model(inputs, outputs, name='U-Net')


if __name__ == '__main__':
    model = build_unet((256, 256, 3), 8)
    model.summary()

