import cv2
import numpy as np
import os

cap = cv2.VideoCapture(1)

# 项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 人脸检测模型路径
face_model_path = 'data\haarcascades\haarcascade_frontalface_default.xml'
# 人眼检测模型路径
eye_model_path = 'data\haarcascades\haarcascade_eye_tree_eyeglasses.xml'
print('项目根目录: ' + root_dir)

face_model = cv2.CascadeClassifier(
    os.path.join(root_dir, face_model_path))
# 加载人眼检测模型
eye_model = cv2.CascadeClassifier(
    os.path.join(root_dir, eye_model_path))

while True:
    ret, frame = cap.read()
    if ret is None:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi_gray = None
    roi_color = None
    faces = face_model.detectMultiScale(gray_frame, 1.1, 8, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # 在检测区域打印人脸坐标与大小
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('face_track', frame)

    eyes = eye_model.detectMultiScale(roi_gray, 1.05, 10, minSize=(30, 30))

    for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        cv2.circle(roi_color, (ex + ew // 2, ey + eh // 2), ew // 2, (255, 0, 0), 1)
        # 在检测区域打印眼睛坐标与大小
        cv2.putText(roi_color, 'Eye', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow('eye_track', roi_color)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
