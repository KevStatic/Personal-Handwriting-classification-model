# scripts/train.py

import os
import pickle
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# -------------------------------
# Load MNIST dataset (digits 0-9)
# -------------------------------
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

# -------------------------------
# Character map
# -------------------------------
char_map = {i: str(i) for i in range(10)}  # digits 0-9

# Create saved_models directory if it doesn't exist
os.makedirs('../saved_models', exist_ok=True)

# Save character map
with open('../saved_models/char_map.pkl', 'wb') as f:
    pickle.dump(char_map, f)

# -------------------------------
# Build CNN model
# -------------------------------
print("Building CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(char_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# Train the model
# -------------------------------
print("Training model...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# -------------------------------
# Save the trained model
# -------------------------------
model_path = '../saved_models/my_handwriting_model.h5'
model.save(model_path)
print(f"Model saved at: {model_path}")
print("Training complete!")
