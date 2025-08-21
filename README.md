# Face Tracker – DNN Version

A deep-learning-based Python tracker for real-time face detection using OpenCV's DNN module.

## Features
- Uses deep neural networks for face detection.
- Better detection accuracy under different conditions.
- Real-time performance (GPU supported if available).

## Prerequisites
- Python 3.6+
- Install dependencies:
  pip install opencv-python numpy

## How to Run
python main.py

Customize runtime parameters or model paths as needed.

## Project Structure
face-tracker-dnn/
├── main.py
└── README.md

## Future Improvements
- Add support for multiple models (SSD, YOLO, RetinaFace).
- Add option to save annotated video output.
- Multi-face tracking and logging.
