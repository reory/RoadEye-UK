import os
import warnings
import logging

# Force TensorFlow to silence errors/warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Block the Python-level logging output from TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np #noqa
import tensorflow as tf #noqa

from sklearn.model_selection import train_test_split #noqa
from tensorflow.keras import layers, models #noqa
from data_loader import load_traffic_data #noqa

# Suppress general Keras/Python user warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Fetch and process data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset")

print("Fetching data using data loader.py...")
X, y = load_traffic_data(DATA_PATH)

# Normalize pixels to 0.0 - 1.0
# X = X / 255.0

# Convert text labels to unique numbers using NumPy
unique_labels = np.unique(y)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label_to_index[label] for label in y])

print(f"Mapped {len(unique_labels)} unique classes to numeric IDs.")

# Shuffle and Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("--Loading pre-trained MobileNetV2 base--")

# Load base model once
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Unfreeze the base model, but freeze everything except the very lasy block
base_model.trainable = True

# MobileNetV2 has 154 layers. We freeze the first 130 and train the rest.
for layer in base_model.layers[:130]:
    layer.trainable = False

# Construct the assembly line
model = models.Sequential()

# Input matches your raw dataset image size
model.add(layers.Input(shape=(224, 224, 3)))

# Data Augmentation
model.add(layers.RandomRotation(factor=0.04))
model.add(layers.RandomZoom(height_factor=0.15, width_factor=0.15))
model.add(layers.RandomBrightness(factor=0.2))
model.add(layers.RandomContrast(factor=0.1))  

# MobileNetV2 preprocessing utility built directly into the graph
model.add(layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input))

# Insert the partially unfrozen powerhouse
model.add(base_model)

# Flatten out the 3D feature maps into a 1D vector cleanly
model.add(layers.GlobalAveragePooling2D())

# Your custom classification head
model.add(layers.Dense(
    128, 
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# Increased to 0.5 to stop the model from memorizing the background details
model.add(layers.Dropout(0.6))

# Output layer matching your 16 unique traffic sign classes
model.add(layers.Dense(len(unique_labels), activation='softmax'))

# Compile the Model 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--📝 Model Summary--")
model.summary()

# Train the Model
print("\n🚀 Starting Model Training..")


lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5, 
    restore_best_weights=True 
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20, 
    batch_size=32, 
    callbacks=[lr_reducer, early_stopping]
)

print("\n✅ Training Complete!")

# Save the trained brain so it can be used later
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "traffic_sign_model.keras")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f" 🤖 Model successfully saved: {MODEL_SAVE_PATH}")