import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
from keras.models import load_model

model = load_model("MyModel.h5")

def predict_image(image):
    img = np.array(image)
    img = img / 255.0  # Chuẩn hóa hình ảnh nếu cần
    img = np.expand_dims(img, axis=0)  # Thêm một chiều cho batch size

    # Dự đoán
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    
    class_names = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot','cauliflower','chilli pepper',
               'corn','cucumber','eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango',
               'onion','orange','paprika','pear','peas','pineapple','pomegranate','potato','raddish','soy_beans',
               'spinach','sweetcorn','sweetpotato','tomato','turnip','watermelon']
    return class_names[class_index]

# Bước 3: Tạo giao diện Gradio
iface = gr.Interface(
    fn=predict_image,  # Hàm dự đoán
    inputs=gr.Image(type="pil"),  # Đầu vào là một hình ảnh
    outputs="text",  # Kết quả trả về là văn bản tên lớp (cat, dog, rabbit)
    title="Animal Classifier",  # Tiêu đề của ứng dụng
    description="Upload an image to classify if it is a cat, dog, or rabbit."  # Mô tả ngắn
)

# Khởi chạy giao diện Gradio
iface.launch()
