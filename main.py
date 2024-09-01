import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import gradio as gr

label = "Warmup...."
n_time_steps = 10


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("NewModel.keras")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
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
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape) 
    results = model.predict(lm_list)
    print("Result = " + str(results)) 
    # Get the index of the highest probability
    max_prob = np.max(results)
    class_index = np.argmax(results, axis=1)[0]
    # Map the index to the corresponding label
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

cap1 = cv2.VideoCapture("test_video.mp4")
lm_list = []
i = 0
warmup_frames = 10
while True:
    success, img = cap1.read()
    if not success:
        print("Failed to read frame or video ended")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect....")
        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []
            img = draw_landmark_on_image(mpDraw, results, img)
    img = draw_class_on_image(label, img)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

