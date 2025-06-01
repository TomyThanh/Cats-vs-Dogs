import os #hiiiiii yolo Ömer
import itertools 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.utils import plot_model
import random

##Visualizing model's predictions with a COMPLEX FUNCTION
def plot_decision_boundary(model, X, y):
    """Plots the decision boundary created by a model predicting on X"""
    X_min, X_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(X_min, X_max, 100),
                         np.linspace(y_min, y_max, 100))

    X_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(X_in)
    y_pred = np.reshape(y_pred, xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), xx.max())
    plt.show()


# Make some functions to reuse MAE and MSE
def mae(y_true, y_pred):
    mae = tf.metrics.MeanAbsoluteError()
    mae.update_state(y_true, tf.squeeze(y_pred))
    result = mae.result().numpy()
    return result

def mse(y_true, y_pred):
    mse = tf.metrics.MeanSquaredError()
    mse.update_state(y_true, tf.squeeze(y_pred))
    result = mse.result().numpy()
    return result


#replicating sigmoid-function
def sigmoid(x):
    return 1/(1+tf.exp(-x))

#replicating relu-function
def relu(x):
    return tf.maximum(0, x)

#CONFUSION MATRIX
def plot_confusion_matrix_pretty(y_true, y_pred, class_names=None, normalize=True, figsize=(10,10), cmap=plt.cm.Blues):
    """
    Zeigt eine formatierte Confusion Matrix mit Optionen für Normalisierung und Klassenbeschriftung.

    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels (kann probabilistisch sein – wird gerundet)
    - class_names: list of class names (default: nummerisch)
    - normalize: bool, ob Matrix normalisiert werden soll
    - figsize: tuple, Größe der Darstellung
    - cmap: Farbkarte
    """

    # Vorhersagen ggf. runden (bei probabilistischen Outputs)
    # Vorhersagen: Wahrscheinlichkeiten zu Klassenlabels machen
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Confusion Matrix berechnen
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    n_classes = cm.shape[0]

    # Klassenlabels
    labels = class_names if class_names else np.arange(n_classes)

    # Plot Setup
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Schwellenwert für Textfarbe
    threshold = (cm.max() + cm.min()) / 2.

    # Zahlen & Prozentwerte eintragen
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        value = f"{cm[i, j]}"
        if normalize:
            value += f"\n({cm_norm[i, j]*100:.1f}%)"
        ax.text(j, i, value,
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=13)

    plt.tight_layout()
    plt.show()


#Picks a random image, plots it and labels it with a prediction and truth label
def plot_random_image(model, images, true_labels, classes):
    i = random.randint(0, len(images))

    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28,28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]
    
    plt.imshow(target_image, cmap=plt.cm.binary)
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label, 100*tf.reduce_max(pred_probs), true_label), 
               color = color) #set the color to green or red based on if prediction is wrong or right
    plt.show()






