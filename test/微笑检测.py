import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)

# 项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 人脸检测模型路径
face_model_path = 'data\haarcascades\haarcascade_frontalface_default.xml'
# 微笑表情模型路径
smile_model_path = 'data\haarcascades\haarcascade_smile.xml'
print('项目根目录: ' + root_dir)

face_model = cv2.CascadeClassifier(
    os.path.join(root_dir, face_model_path))
smile_model = cv2.CascadeClassifier(
    os.path.join(root_dir, smile_model_path))

while True:
    ret, frame = cap.read()
    if ret is None:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray_frame, 1.3, 12, minSize=(100, 100))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Face size: {w}x{h}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        roi_gray_face = gray_frame[y:y + h, x:x + w]
        smiles = smile_model.detectMultiScale(roi_gray_face, 1.8, 40, minSize=(80, 40))

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 1)
            if len(smiles) > 0:
                cv2.putText(frame, f'Smile size: {sw}x{sh}', (x+sx, y+sy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
