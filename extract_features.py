# installing dependencies
!pip install ultralytics opencv-python-headless numpy scikit-learn matplotlib seaborn
!pip install tensorflow

# import necessary packages
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dropout, TimeDistributed, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# compute and extract features in preprocessing
yolo = YOLO('yolov10n.pt')  # Load YOLOv10-nano

def get_intersection_roi(boxes):
    if len(boxes) < 2:
        return None
    x1 = max(b[0] for b in boxes)
    y1 = max(b[1] for b in boxes)
    x2 = min(b[2] for b in boxes)
    y2 = min(b[3] for b in boxes)
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]

def compute_optical_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                        pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def extract_hog(img):
    # Return only the HOG features (not the visual image)
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=True, channel_axis=None)
    return features



