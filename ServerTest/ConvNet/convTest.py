import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keras.datasets import mnist, fashion_mnist
import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf



# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = kb.utils.to_categorical(y_train, 10)
y_test = kb.utils.to_categorical(y_test, 10)

model = kb.Sequential([
    
    kb.layers.Conv2D(32, (3, 3), activation='relu', padding = "same", input_shape=(28, 28, 1)),
    kb.layers.Conv2D(64, (3, 3), activation='relu', padding = "same"),
    kb.layers.Conv2D(128, (3, 3), activation='relu', padding = "same"),
    kb.layers.MaxPooling2D((2, 2)),

    kb.layers.Conv2D(256, (3, 3), activation='relu', padding = "same"),
    kb.layers.Conv2D(512, (3, 3), activation='relu', padding = "same"),
    kb.layers.MaxPooling2D((2, 2)),
    kb.layers.Flatten(),

    kb.layers.Dense(100, activation='relu'),
    kb.layers.Dense(50, activation='relu'),
    kb.layers.Dense(10, activation='softmax'),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, validation_data = (x_test, y_test))