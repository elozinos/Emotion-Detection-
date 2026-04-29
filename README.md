Emotion Detection System
An application designed to identify and classify human emotions from facial expressions using computer vision and machine learning techniques.

Features
Real-time Detection: Capture and analyze emotions via webcam or video feed.

Multi-Class Classification: Recognizes emotions including Happy, Sad, Angry, Surprised, Neutral, and Fear.

Image Preprocessing: Automated grayscale conversion, resizing, and normalization.

Deep Learning Architecture: Optimized for high accuracy and performance.

Tech Stack
Language: Python

Libraries: OpenCV, TensorFlow/Keras, NumPy, Pandas, and Matplotlib.

Project Structure
models/ : Contains pre-trained model files and weights.

data/ : Directory for training and testing datasets.

src/ : Core source code for training and inference.

requirements.txt : List of necessary dependencies for the project.

Installation and Setup
Clone the repository:
git clone https://github.com/elozinos/Emotion-Detection-.git
cd Emotion-Detection-

Install dependencies:
pip install -r requirements.txt

Run the application:
python src/detect.py

Dataset
This project utilizes the FER2013 (Facial Expression Recognition 2013) dataset, which consists of 48x48 pixel grayscale images categorized into seven distinct emotion classes.

Future Development
Integration of a web-based dashboard using Streamlit or Flask.

Support for batch processing of video files.

Enhanced robustness for varied lighting conditions and head poses.
