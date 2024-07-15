import os
import dlib
import cv2
import numpy as np

# 项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 图片路径
img_path = 'data\\images\\lxy01.jpg'


def CannyOperate(val, sigma=50):
    lowThreshold = val  # 最小阈值
    kernel_size = 3  # 边缘检测的核大小
    sigma_radius = sigma / 100.0  # 高斯滤波的sigma值
    ratio_canny = 3  # 边缘检测的比例
    # 高斯滤波
    detected_edges = cv2.GaussianBlur(img_face_gray, (kernel_size, kernel_size), sigma_radius)
    # Canny边缘检测 参数：图像、最小阈值（范围：0-255）、最大阈值（范围：0-255）、核大小
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio_canny, kernel_size)

    # 与运算，将边缘检测结果与原图进行合并 参数：原图、掩模、掩膜(意思是将边缘检测结果的非黑色区域变成白色，其他区域变成黑色)
    dst = cv2.bitwise_and(img_face_gray, img_face_gray, mask=detected_edges)
    # 显示边缘检测结果
    cv2.imshow("Canny-demo", dst)


# 加载dlib预训练的人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(root_dir, "shape_predictor_68_face_landmarks.dat"))

# 读取图片
img = cv2.imread(os.path.join(root_dir, img_path))

# 等比例缩放图片
img_width = 600
img_height = int(img_width * img.shape[0] / img.shape[1])
img = cv2.resize(img, (img_width, img_height))
img_face = None
img_face_gray = None
# 灰度化图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray, 1)
for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img_face = img[y1:y2, x1:x2]
    img_face_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)

# 边缘检测
cv2.namedWindow("Canny-demo")
# 创建滑动条
cv2.createTrackbar("lowThreshold", "Canny-demo", 0, 100, CannyOperate)
cv2.createTrackbar("Sigma", "Canny-demo", 0, 100, CannyOperate)
# 初始调用一次CannyOperate，以免lowThreshold为0时，无法显示边缘检测效果
CannyOperate(0, 0)

# 显示原图和人脸
cv2.imshow("original", img)
cv2.imshow("face", img_face)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
