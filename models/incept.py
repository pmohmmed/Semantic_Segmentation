from tensorflow.keras import layers
from tensorflow.keras.models import Model

def inception_block(x, filters):
    branch1 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(x)

    branch2 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
    branch2 = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(branch2)

    branch3 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
    branch3 = layers.Conv2D(filters, (5, 5), padding="same", activation="relu")(branch3)

    branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch4 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(branch4)

    output = layers.Concatenate()([branch1, branch2, branch3, branch4])
    return output


def build_inception(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    skip1 = x 

    x = layers.MaxPooling2D((2, 2))(x)

    x = inception_block(x, 64)
    skip2 = x

    x = layers.MaxPooling2D((2, 2))(x)

    x = inception_block(x, 128)
    skip3 = x

    x = layers.MaxPooling2D((2, 2))(x)
    x = inception_block(x, 256)

    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x, skip3])
    x = inception_block(x, 128)

    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x, skip2])
    x = inception_block(x, 64)

    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x, skip1])

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x)

    return Model(inputs, outputs)


