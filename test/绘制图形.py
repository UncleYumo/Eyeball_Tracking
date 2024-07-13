import cv2
import numpy as np
import os

def face_detection(img):
    slices = []
    # 提取人脸
    face_cascade = cv2.CascadeClassifier(
        os.path.join(project_path, 'data\\haarcascades\\haarcascade_frontalface_default.xml'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, text=f'Face: {w}x{h}', org=(x, y - 5), fontScale=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 255), thickness=1)
        slices.append(img[y:y + h, x:x + w])
    return slices

def eye_detection(img):
    slices = []
    # 提取眼睛
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(project_path, 'data\\haarcascades\\haarcascade_eye.xml'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text=f'Eye: {w}x{h}', org=(x, y - 5), fontScale=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 255), thickness=1)
        slices.append(img[y:y + h, x:x + w])
    return slices

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(project_path, 'data\\images\\uncleyumo01.jpg')

# 判断是否有图片路径
if not img_path:
    print('No image path provided')
    exit()
else:
    print('Image path:', img_path)

# 读取图片，并进行缩放
width = 1000
img = cv2.imread(img_path)
img = cv2.resize(img, (width, img.shape[0] * width // img.shape[1]))
original_size = img.shape
cv2.putText(img, text=f'Original size: {original_size}', org=(0, 20), fontScale=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            color=(0, 0, 255), thickness=1)


# 眼睛检测
slices = eye_detection(img)
for i, slice in enumerate(slices):
    cv2.imshow(f'Eye {i}', slice)

# 显示图片
cv2.imshow('Image', img)


key = cv2.waitKey(0)  # 等待按键按下
if key == 27:
    cv2.destroyAllWindows()
