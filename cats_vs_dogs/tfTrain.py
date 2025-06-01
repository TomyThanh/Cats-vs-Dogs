import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tfFUNCTIONS import *
import tensorflow_datasets as tfds

def build_model():
    model = tf.keras.Sequential([
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model
