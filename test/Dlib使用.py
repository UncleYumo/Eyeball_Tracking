import dlib
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 加载dlib预训练的人脸检测器
detector = dlib.get_frontal_face_detector()
# 加载dlib预训练的人脸关键点检测器
predictor = dlib.shape_predictor("E:\Dev_work\Python_Dev\Pycharm_Project\eye_track_study\shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if ret is None:
        break

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用detector进行人脸检测
    faces = detector(gray, 0)   # 0表示使用默认的阈值, 1表示使用更高的阈值（1为上采样，0为下采样）

    # 绘制矩形框
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, center=((x1+x2)//2, (y1+y2)//2), radius=2, color=(0, 0, 255), thickness=2)

        # 使用predictor进行人脸关键点检测
        shape = predictor(gray, face)
        for i in range(68):
            cv2.circle(frame, center=(shape.part(i).x, shape.part(i).y), radius=2, color=(255, 0, 0), thickness=2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
