import numpy as np


def infer(images:np.array, model):
    
    # check input shape
    if images.shape[1:] !=  model.input_shape[1:]:
        print(f'|!| Expected input shape: {model.input_shape[1:]}')
        return []
    else:
        # predict
        predicted_labels = model.predict(images)

        # select highist class for each pixel
        encoded_labels = np.array([np.argmax(label, axis=-1)
                                   for label in predicted_labels])

        return encoded_labels

