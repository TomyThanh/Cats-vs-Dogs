import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tfFUNCTIONS import *
import tensorflow_datasets as tfds
from tfTrain import build_model

# Verzeichnis, in dem deine Bilder gespeichert sind
data_dir = r"C:\Unkram123\project_folder"  # Dein Pfad zu den Bildern

# Daten laden
train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,  # Pfad zu deinem Trainingsdatensatz
    image_size=(224, 224),  # Bildgröße
    batch_size=32,
    validation_split=0.2,  # 20% für Validierung
    subset="training",  # Trainingsdaten
    seed=42,
    label_mode="int"  # Labels als Ganzzahlen
)

val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,  # Pfad zu deinen Trainingsdaten (kann der gleiche wie oben sein)
    image_size=(224, 224),  # Bildgröße
    batch_size=32,
    validation_split=0.2,  # 20% für Validierung
    subset="validation",  # Validierungsdaten
    seed=42,
    label_mode="int"  # Labels als Ganzzahlen
)

# Optional: Normalisierung der Bilder
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

#Training
model = build_model()
history = model.fit(train_data, validation_data = val_data, epochs = 50)


























































































