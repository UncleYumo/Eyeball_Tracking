import os

import dlib
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# 项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载dlib预训练的人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(root_dir, "shape_predictor_68_face_landmarks.dat"))

while True:
    ret, frame = cap.read()
    if ret is None:
        break

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用detector进行人脸检测
    faces = detector(gray, 0)   # 0表示使用默认的阈值, 1表示使用更高的阈值（1为上采样，0为下采样）
    face_img = None

    # 绘制矩形框
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # 计算人脸特征点
        shape = predictor(gray, face)
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 0, 255), 1)
        area_eye_np = np.array([
                (shape.part(17).x, shape.part(17).y),
                (shape.part(26).x, shape.part(26).y),
                (shape.part(15).x, shape.part(15).y),
                (shape.part(1).x, shape.part(1).y)
        ], np.int32)

        # 绘制眼睛轮廓
        cv2.polylines(frame, [area_eye_np], True, (0, 255, 0), 1)

        # 裁剪人脸区域
        face_img = frame[y1:y2, x1:x2]
        # 显示裁剪出的人脸部分图像
        if face_img is not None:
            cv2.imshow("face", face_img)

    cv2.imshow("frame", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
