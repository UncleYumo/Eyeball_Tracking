import os
import dlib
import cv2
import numpy as np

# 项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 图片路径
img_path = 'data\\images\\lxy01.jpg'

# 加载dlib预训练的人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(root_dir, "shape_predictor_68_face_landmarks.dat"))

# 读取图片
img = cv2.imread(os.path.join(root_dir, img_path))

# 等比例缩放图片
img_width = 2000
img_height = int(img_width * img.shape[0] / img.shape[1])
img = cv2.resize(img, (img_width, img_height))

# 灰度化图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray, 1)
face_img = None

for face in faces:

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    # 截取人脸区域
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    face_img = img[y1:y2, x1:x2]

    # 计算人脸的68个特征点坐标
    shape = predictor(gray, face)
    # 绘制68个特征点
    for i in range(68):
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 1)
        cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("face", face_img)

cv2.imshow("original", img)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()