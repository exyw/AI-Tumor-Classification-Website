import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

# preprocess the data
dataset_dir = '/Users/emmaxu/downloads/brain-mri-dataset'
image_size = (128, 128)

# map the classes and labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_labels = {cls: idx for idx, cls in enumerate(classes)}

images = []
labels = []
image_paths = []

# iterate through each class folder
for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    if not os.path.isdir(class_dir):
        print(f"Warning: Folder {class_dir} not found. Skipping")
        continue

    # process each image
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image {image_path}. Skipping")
                continue

            # resize the images and normalise the pixel values to train the model
            # faster and more efficiently
            img_resized = cv2.resize(img, image_size)
            img_normalised = img_resized / 255.0
            images.append(img_normalised)
            labels.append(class_labels[cls])
            image_paths.append(image_path)

        except Exception as e:
            print(f"Erorr processing image {image_path}: {e}")
            continue

    print("one folder done\n")

# convert lists to NumPy arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print(f"Processed {len(images)} images.")

# splitting the data, where x is the image, y is the label, and paths is the filename
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(images, labels, image_paths, test_size=0.2, random_state=71)
X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(X_train, y_train, paths_train, test_size=0.2, random_state=71)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Testing set: {len(X_test)} samples")

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    InputLayer(shape=(128, 128, 3)),
    Conv2D(16, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.25),
    Dense(4, activation='softmax', kernel_regularizer=regularizers.l2(0.001))  # 4 classes: glioma, meningioma, no tumor, pituitary
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train the model
history = model.fit(
    X_train,              # Training data
    y_train,              # Training labels
    batch_size=16,        # Number of samples per gradient update
    epochs=30,            # Number of epochs to train
    validation_data=(X_val, y_val),  # Validation data
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=16)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model.save('model1.h5')  # Save the model to a file