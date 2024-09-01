from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("NewModel.keras")
cap = cv2.VideoCapture("...")
lm_list = []
n_time_steps = []

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.resize(frame, (224,224))
    frame = frame / 255.0
    
    landmarks = extrac