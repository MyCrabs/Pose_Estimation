import cv2
import gradio as gr

def capture_video():
    cap = cv2.VideoCapture(0)  # Mở webcam mặc định
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
        yield frame  # Trả về khung hình cho Gradio xử lý

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=capture_video,  # Hàm xử lý khung hình từ webcam
    inputs=None,  # Không cần đầu vào từ người dùng
    outputs=gr.Image(type="pil"),  # Kết quả là một hình ảnh sử dụng Gradio mới
    live=True  # Cho phép cập nhật hình ảnh liên tục
)

# Khởi chạy giao diện Gradio
iface.launch()
