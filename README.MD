# Face Detection App

A Streamlit-based application for face detection using OpenCV's Haar Cascade classifiers. This app provides multiple tools for detecting faces in images and real-time video streams.

## Features

- **Welcome Page**: Introduction to the app with navigation sidebar.
- **Face Detection on Images**: Upload an image (JPG, JPEG, PNG) and detect faces with bounding boxes.
- **Real-Time Face Detection**: Use your webcam to detect faces in live video with face count display.
- **Face + Eyes Detection**: Upload an image to detect faces and eyes with color-coded rectangles.

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd facedetection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```
streamlit run main.py
```

Navigate through the app using the sidebar:

- Select "Face Detection" for image-based detection.
- Select "Face Detection App" for real-time webcam detection.
- Select "Face + Eyes Detection" for advanced feature detection.

## Requirements

- Python 3.7+
- streamlit
- opencv-python-headless
- numpy
- Pillow

## How It Works

The app uses OpenCV's Haar Cascade classifiers for detecting facial features:

- `haarcascade_frontalface_default.xml` for face detection.
- `haarcascade_eye.xml` for eye detection.

## License

This project is open-source. Feel free to use and modify as needed.
