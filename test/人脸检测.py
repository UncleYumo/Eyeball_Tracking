import cv2
import numpy as np

cap = cv2.VideoCapture(1)

# 加载人脸检测分类器
face_cascade = cv2.CascadeClassifier('E:\Dev_work\Python_Dev\Pycharm_Project\eye_track_study\data\haarcascades\haarcascade_frontalface_default.xml')


def face_detect(_frame):
    _frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _frame_faces = []
    # 人脸检测
    faces = face_cascade.detectMultiScale(_frame_gray, 1.1, 10, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        _frame_faces.append(_frame[y:y+h, x:x+w])
        cv2.putText(_frame, f"Face{len(_frame_faces)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    return _frame_faces, _frame_gray


while True:
    ret, frame = cap.read()
    if ret is None:
        break

    key = cv2.waitKey(25)
    if key == 27:
        break

    frame_faces, frame_gray = face_detect(frame)
    cv2.imshow("frame", frame)
    cv2.imshow("frame_gray", frame_gray)

    if len(frame_faces) > 0:
        cv2.imshow("face1", frame_faces[0])

    cv2.moveWindow("frame", 200, 200)
    cv2.moveWindow("frame_gray", 200+frame.shape[1], 200)

# 销毁所有窗口并释放摄像头
cap.release()
cv2.destroyAllWindows()
