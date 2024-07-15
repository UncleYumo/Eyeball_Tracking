import math

import cv2
import dlib
import numpy as np
import os


# 彩色打印 默认青色字体，无底色
# red: 31, green: 32, yellow: 33, blue: 34, purple: 35, cyan: 36, white: 37
def color_print(msg, color="cyan"):
    if color == "red":
        color_code = "\033[31m"
    elif color == "green":
        color_code = "\033[32m"
    elif color == "yellow":
        color_code = "\033[33m"
    elif color == "blue":
        color_code = "\033[34m"
    elif color == "purple":
        color_code = "\033[35m"
    elif color == "cyan":
        color_code = "\033[36m"
    elif color == "white":
        color_code = "\033[37m"
    else:
        color_code = "\033[37m"
    print(color_code + msg + "\033[0m")


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
def draw_face_landmarks(img, face_landmarks, is_show_index=False):
    for i in range(68):
        x = face_landmarks.part(i).x
        y = face_landmarks.part(i).y
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)  # 绘制人脸关键点
        if is_show_index:
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1)  # 绘制人脸关键点索引
    # return img


# 获取眼部矩形区域四角坐标（已弃用）
def get_eye_location(face_landmarks):
    left_top_x = face_landmarks.part(0).x
    left_top_y = face_landmarks.part(18).y
    right_top_x = face_landmarks.part(16).x
    right_top_y = face_landmarks.part(25).y
    left_bottom_x = face_landmarks.part(2).x
    left_bottom_y = face_landmarks.part(2).y
    right_bottom_x = face_landmarks.part(16).x
    right_bottom_y = face_landmarks.part(14).y
    eye_rac_location = (
        (left_top_x, left_top_y),  # eye_rac_location[0][0] and [0][1]
        (right_top_x, right_top_y),  # eye_rac_location[1][0] and [1][1]
        (left_bottom_x, left_bottom_y),  # eye_rac_location[2][0] and [2][1]
        (right_bottom_x, right_bottom_y)  # eye_rac_location[3][0] and [3][1]
    )
    return eye_rac_location


# 获取眼部矩形区域四角坐标
def get_eye_location_PLUS(face_landmarks):
    top_y = min(face_landmarks.part(17).y, face_landmarks.part(26).y)
    bottom_y = max(face_landmarks.part(1).y, face_landmarks.part(15).y)
    left_x = face_landmarks.part(0).x
    right_x = face_landmarks.part(16).x

    eye_rac_location = (
        (left_x, top_y),  # eye_rac_location[0][0] and [0][1]
        (right_x, top_y),  # eye_rac_location[1][0] and [1][1]
        (left_x, bottom_y),  # eye_rac_location[2][0] and [2][1]
        (right_x, bottom_y)  # eye_rac_location[3][0] and [3][1]
    )
    return eye_rac_location


# 获取左右眼的眼球中心坐标与眼球半径
def get_pupil_location(face_landmarks):
    left_pupil_x_left = face_landmarks.part(45).x
    left_pupil_y_left = face_landmarks.part(45).y

    left_pupil_x_right = face_landmarks.part(42).x
    left_pupil_y_right = face_landmarks.part(42).y

    left_pupil_radius = math.sqrt((left_pupil_x_left - left_pupil_x_right) ** 2 +
                                  (left_pupil_y_left - left_pupil_y_right) ** 2) / 2
    left_pupil_center = (int((left_pupil_x_left + left_pupil_x_right) / 2),
                         int((left_pupil_y_left + left_pupil_y_right) / 2))

    right_pupil_x_left = face_landmarks.part(39).x
    right_pupil_y_left = face_landmarks.part(39).y

    right_pupil_x_right = face_landmarks.part(36).x
    right_pupil_y_right = face_landmarks.part(36).y

    right_pupil_radius = math.sqrt((right_pupil_x_left - right_pupil_x_right) ** 2 +
                                   (right_pupil_y_left - right_pupil_y_right) ** 2) / 2
    right_pupil_center = (int((right_pupil_x_left + right_pupil_x_right) / 2),
                          int((right_pupil_y_left + right_pupil_y_right) / 2))

    return left_pupil_center, left_pupil_radius, right_pupil_center, right_pupil_radius


def draw_eye_location(frame, eye_location):
    if eye_location is None:
        return None
    cv2.rectangle(frame,
                  (eye_location[0][0], eye_location[0][1]),
                  (eye_location[3][0], eye_location[3][1]),
                  (255, 0, 0),
                  1)
    # return frame


# 获取眼部区域图片
def get_eye_image(frame, eye_location):
    if eye_location is None:
        return None
    rec_top_y = math.ceil((eye_location[0][1] + eye_location[1][1]) / 2)
    rec_bottom_y = math.ceil((eye_location[2][1] + eye_location[3][1]) / 2)
    rec_left_x = math.ceil((eye_location[0][0] + eye_location[2][0]) / 2)
    rec_right_x = math.ceil((eye_location[1][0] + eye_location[3][0]) / 2)
    eye_image = frame[rec_top_y:rec_bottom_y, rec_left_x:rec_right_x]
    return eye_image


def draw_pupil_location(frame, left_pupil_center, left_pupil_radius, param):
    if left_pupil_center is None:
        return None
    cv2.circle(frame,
               left_pupil_center,
               int(left_pupil_radius),
               param,
               1)
    # return frame
