import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy as np

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

# 初始化摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    # 从摄像头捕获一帧图像
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    # 处理捕获到的图像
    processed_frame = process_frame(frame)
    
    # 显示处理后的图像
    cv2.imshow('Face Recognition', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()
