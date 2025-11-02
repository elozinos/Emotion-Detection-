import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. Load dataset from the .npz file
data = np.load("fer_dataset.npz")
trainX, trainY = data["trainX"], data["trainY"]
testX, testY = data["testX"], data["testY"]

# 2. Normalize image pixel values (0–255 → 0–1)
trainX = trainX / 255.0
testX = testX / 255.0

# 3. Reshape to (image_height, image_width, channels) → (48, 48, 1)
trainX = trainX.reshape(-1, 48, 48, 1)
testX = testX.reshape(-1, 48, 48, 1)

# 4. Convert labels to one-hot (e.g. 3 → [0,0,0,1,0,0,0])
trainY = to_categorical(trainY, num_classes=7)
testY = to_categorical(testY, num_classes=7)

# 5. Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# 6. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Save only the best model during training
checkpoint = ModelCheckpoint("face_emotionModel.h5", monitor='val_accuracy', save_best_only=True, mode='max')

# 8. Train the model
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=64, callbacks=[checkpoint])

print("✅ Training complete! Best model saved as 'face_emotionModel.h5'")
