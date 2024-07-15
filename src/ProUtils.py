import cv2
import dlib
import numpy as np
import os


# 彩色打印 默认青色字体，无底色
def color_print(msg, color=32, bg_color=None):   # 32:青色 31:红色 33:黄色 34:蓝色 35:紫色 36:天蓝色 37:白色
    if bg_color is None:
        print('\033[1;%dm%s\033[0m' % (color, msg))
    else:
        print('\033[1;%d;%dm%s\033[0m' % (color, bg_color, msg))


# 检查文件是否存在
def check_whether_the_file_exists(file_path):
    if not os.path.exists(os.path.join(root_path, file_path)):
        print(f"{file_path}文件不存在，请检查文件路径！")
        exit()


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
detector = dlib.get_frontal_face_detector()  # 加载dlib人脸检测器
check_whether_the_file_exists("shape_predictor_68_face_landmarks.dat")  # 检查人脸关键点检测器文件是否存在
predictor = dlib.shape_predictor(os.path.join(root_path, "shape_predictor_68_face_landmarks.dat"))  # 加载dlib人脸关键点检测器


# 间隔指定时间后等待指定按键
def wait_for_key(time=25, key=27):
    key_get = cv2.waitKey(time)
    if key_get == key:
        return True
    return False


# 脸大优先人脸区域矩形坐标算法
def get_face_location(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    faces = detector(img_gray, 1)  # 检测人脸

    # 判断是否检测到人脸
    if len(faces) <= 0:
        return None, None

    # 遍历faces，只保留面积最大的人脸
    max_area = 0
    max_face = None
    for face in faces:
        area = (face.right() - face.left()) * (face.bottom() - face.top())
        if area > max_area:
            max_area = area
            max_face = face

    face_location = (max_face.left(), max_face.top(),
                     max_face.right() + 1, max_face.bottom() + 1)  # +1实现矩形区间闭合

    return face_location, max_face


# 绘制人脸区域矩形
def draw_face_location(img, face_location):
    x1, y1, x2, y2 = face_location
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 绘制人脸区域矩形
    # return img


# 根据人脸坐标获取人脸关键点，返回68个关键点坐标和人脸区域图片
def get_face_landmarks(img, face_dlib):
    face_landmarks = predictor(img, face_dlib)  # 获取人脸关键点
    return face_landmarks


# 绘制人脸关键点
def draw_face_landmarks(img, face_landmarks):
    for i in range(68):
        x = face_landmarks.part(i).x
        y = face_landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # 绘制人脸关键点
    # return img


# 获取眼部矩形区域坐标
def get_eye_region(face_landmarks):
    eye_region = []
    x1 = face_landmarks.part(17).x
    y1 = face_landmarks.part(18).y
    x2 = face_landmarks.part(26).x
    y2 = face_landmarks.part(1).y
    eye_region.append((x1, y1, x2, y2))
    return eye_region


# 获取眼部图片区域
def get_eyeImage_region(img, eye_region):
    x1 = eye_region[0]
    y1 = eye_region[1]
    x2 = eye_region[2]
    y2 = eye_region[3]
    eye_images = img[y1:y2, x1:x2]
    return eye_images


