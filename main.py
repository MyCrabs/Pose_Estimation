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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("NewModel.keras")
cap = cv2.VideoCapture(0)


def make_landmark_timestep(results):
    c_lm = []
    for _, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mp_draw, results, img):
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    for _, lm in enumerate(results.pose_landmarks.landmark):
        h, w, _ = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def cal_fps(frame_count, start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time
    return str(int(fps))


def draw_class_on_image(label, accuracy, fps, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 30)
    corner_of_accuracy = (10, 110)
    corner_of_fps = (10, 70)
    font_scale = 1
    font_color = (0, 165, 255)
    thickness = 2
    line_type = 2

    cv2.putText(img, label,
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                thickness,
                line_type)
    cv2.putText(img, f"FPS: {fps}",
                corner_of_fps, font, font_scale, font_color, thickness, line_type)
    cv2.putText(img, f"Accuracy: {accuracy:.2f}%",
                corner_of_accuracy,
                font,
                font_scale,
                font_color,
                thickness,
                line_type)
    return img


def detect(model, lm_list):
    global label
    global accuracy
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print("Result = " + str(results))
    max_prob = np.max(results)
    class_index = np.argmax(results, axis=1)[0]
    if max_prob < 0.9:
        label = "UNKNOWN POSE"
        accuracy = 0.00
    else:
        accuracy = max_prob * 100
        if class_index == 0:
            label = "BUTT KICKS"
        elif class_index == 1:
            label = "HIGH KNEES"
        elif class_index == 2:
            label = "JUMPING JACKS"
        elif class_index == 3:
            label = "UNKNOWN POSE"
    return label, accuracy


def capture_video():
    lm_list = []
    i = 0
    warmup_frames = 20
    label = "UNKNOWN POSE"
    accuracy = 0.0
    fps = "0"
    frame_count = 0
    start_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame or video ended.")
            break
        frame_count += 1
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        i += 1

        if i > warmup_frames:
            print("Starting ....")
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    label, accuracy = detect(model, lm_list)
                    lm_list = []
                img = draw_landmark_on_image(mp_draw, results, img)
        if time.time() - start_time >= 1:
            fps = cal_fps(frame_count, start_time)
            frame_count = 0
            start_time = time.time()
        img = draw_class_on_image(label, accuracy, fps, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield img


def process_video(path_video):
    capture = cv2.VideoCapture(f"{path_video}")
    lm_list = []
    i = 0
    warmup_frames = 20
    label = "UNKNOWN POSE"
    fps = "0"
    frame_count = 0
    start_time = time.time()
    accuracy = 0.0

    while True:
        success, img = capture.read()
        if not success:
            print("Failed to read frame or video ended.")
            break
        frame_count += 1
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        i += 1

        if i > warmup_frames:
            print("Starting ....")
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    label, accuracy = detect(model, lm_list)
                    lm_list = []
                img = draw_landmark_on_image(mp_draw, results, img)
        if time.time() - start_time >= 1:
            fps = cal_fps(frame_count, start_time)
            frame_count = 0
            start_time = time.time()
        img = draw_class_on_image(label, accuracy, fps, img)
        cv2.imshow("Video", img)
        if cv2.waitKey(1) == ord('q'):
            break


with gr.Blocks(css=".camera-output {height: 400px;}") as demo:
    with gr.Tabs():
        with gr.TabItem("Camera Pose Estimation"):
            with gr.Column():
                camera_output = gr.Image(label="Camera Output", elem_classes="camera-output")
                capture_button = gr.Button("Start Camera")

        with gr.TabItem("Video Pose Estimation"):
            with gr.Column():
                video_upload = gr.File(label="Upload your video")
                output_text = gr.Textbox(label="Predicted Pose")
                predict_button = gr.Button("Run Pose Estimation")

    capture_button.click(capture_video, outputs=camera_output)
    predict_button.click(process_video, inputs=video_upload, outputs=output_text)

demo.launch()
