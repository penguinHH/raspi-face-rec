import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from picamera2 import Picamera2, Preview

def process_frame(frame):
    # 将捕获到的BGR图像转换为RGB图像，因为face_recognition库需要RGB图像
    rgb_frame = frame[:, :, ::-1]

    # 找到图像中所有面部的所有面部特征
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    
    print("I found {} face(s) in this frame.".format(len(face_landmarks_list)))
    
    # 创建一个PIL imagedraw对象，以便我们可以在图像上绘制
    pil_image = Image.fromarray(rgb_frame)
    d = ImageDraw.Draw(pil_image)
    
    for face_landmarks in face_landmarks_list:
        # 打印此图像中每个面部特征的位置
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
        
        # 在图像中用线条描绘出每个面部特征！
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5)
    
    # 将PIL图像转换回OpenCV图像并返回
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Initialize Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)
picam2.start_preview()  # Correct method to start the preview
picam2.start()

try:
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Convert the frame to RGB as face_recognition expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        processed_frame = process_frame(rgb_frame)

        # Display the processed frame using OpenCV
        cv2.imshow('Face Recognition', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    picam2.stop_preview()  # Correct method to stop the preview
    picam2.stop()
    cv2.destroyAllWindows()