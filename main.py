import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
#
dataset_path = "Dataset"

train_dir = os.path.join(dataset_path, "Train")
val_dir = os.path.join(dataset_path, "Validation")
test_dir = os.path.join(dataset_path, "Test")

image_size = (150, 150)  # Resize images to 150x150
batch_size = 32

print("Started.....")
train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)
print("Image Generator Done")
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode="binary"
)
print("Train Generater")
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size, class_mode="binary"
)
print("Val Generater")

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode="binary", shuffle=False
)
print("Test Generater")

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification (Real vs Fake)
])

print("Model")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
print("Compiled")

epochs = 10

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)
print("Training")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("real_fake_classifier_200_data.h5")
print("Model Saved......")
model = tf.keras.models.load_model("real_fake_classifier_200_data.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    prediction = model.predict(img)
    return "Real" if prediction[0][0] > 0.5 else "Fake"

# Example usage:
image_path = "fake_911.jpg"
result = predict_image(image_path)
print(f"Prediction: {result}")

# plt.plot(history.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training Performance")
# plt.show()
