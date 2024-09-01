import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
import time

# Setup
label = "Warmup...."
n_time_steps = 10

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("NewModel3.keras")
cap = cv2.VideoCapture(0)


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img

def cal_fps(frame_count, start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time
    return str(int(fps))

def draw_class_on_image(label, fps, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    CornerOfFPS = (10, 70)
    fontScale = 1
    fontColor = (0, 165, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.putText(img, f"FPS:{fps}",
                CornerOfFPS, font, fontScale, fontColor, thickness, lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape) 
    results = model.predict(lm_list)
    print("Result = " + str(results)) 
    max_prob = np.max(results)
    class_index = np.argmax(results, axis=1)[0]
    if max_prob < 0.9:
        label == "NONE"
    else:
        if class_index == 0:
            label = "BUTT KICKS"
        elif class_index == 1:
            label = "HIGH KNEES"
        elif class_index == 2:
            label = "JUMPING JACKS" 
    return label

def capture_video():
    lm_list = []
    i = 0
    warmup_frames = 20
    label = "NONE"
    fps ="0"
    frame_count = 0
    start_time = time.time()
    while True:
        success, img = cap.read()
        frame_count += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        i = i + 1
        if i > warmup_frames:
            print("Starting ....")
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    label = detect(model, lm_list)
                    lm_list = []
                img = draw_landmark_on_image(mpDraw, results, img)
        if time.time() - start_time >= 1:
            fps = cal_fps(frame_count, start_time)
            frame_count = 0
            start_time = time.time()
        img = draw_class_on_image(label, fps, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield img
        
def process_video(path_video):
    capture = cv2.VideoCapture(f"{path_video}")
    lm_list = []
    i = 0
    warmup_frames = 20
    label = "NONE"
    fps = "0"
    frame_count = 0
    start_time = time.time()

    while True:
        success, img = capture.read()
        if not success:
            print("Failed to read frame or video ended.")
            break
        
        frame_count += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        i += 1
        
        if i > warmup_frames:
            print("Starting ....")
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    label = detect(model, lm_list)
                    lm_list = []
                img = draw_landmark_on_image(mpDraw, results, img)
        
        if time.time() - start_time >= 1:
            fps = cal_fps(frame_count, start_time)
            frame_count = 0
            start_time = time.time()

        img = draw_class_on_image(label, fps, img)
        cv2.imshow("Video", img)

        
iface = gr.Interface(
    fn=capture_video,
    inputs=None,
    outputs=gr.Image(type="pil"),
    live=True
)

#iface.launch()
process_video("test_video.mp4")