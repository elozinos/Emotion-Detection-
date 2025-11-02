import os
import numpy as np
from PIL import Image

# Path to your dataset directory
data_dir = "data"  # Make sure your dataset folder is named 'data'

# Get list of emotion folders inside 'train' folder
emotions = os.listdir(os.path.join(data_dir, "train"))

def load_data(folder):
    images = []
    labels = []
    for label, emotion in enumerate(emotions):
        emotion_folder = os.path.join(folder, emotion)
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            try:
                img = Image.open(img_path).convert('L')  # convert to grayscale
                img = img.resize((48, 48))              # resize to 48x48
                images.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load training and testing data
trainX, trainY = load_data(os.path.join(data_dir, "train"))
testX, testY = load_data(os.path.join(data_dir, "test"))

# Save data to .npz file
np.savez_compressed("fer_dataset.npz", trainX=trainX, trainY=trainY, testX=testX, testY=testY)

print("✅ Conversion complete! Saved as 'fer_dataset.npz'")
