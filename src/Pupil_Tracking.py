# 本文件实现了瞳孔追踪算法的实现，并提供了一些可视化的工具。
import cv2
import dlib
import numpy as np
import os
import ProUtils as puts
import threading

CAMERA_TYPE = 0
WAIT_TIME = 16  # 等待时间（秒）


def main(wait_time=WAIT_TIME, camera_type=CAMERA_TYPE):
    cap = cv2.VideoCapture(camera_type)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FPS, 30)   # 设置帧率

    frames_num = 0

    while True:
        ret, frame = cap.read()  # 读取一帧图像
        if not ret:
            puts.color_print("摄像头打开失败", "red")
            break

        # 获取计算的人脸位置
        face_location, face_dlib = puts.get_face_location(frame)
        if not face_location:
            puts.color_print(f"未检测到人脸 --> {frames_num}", "yellow")
            continue

        # 在frame上绘制人脸位置
        # print("人脸位置：", face_location)
        # puts.draw_face_location(frame, face_location)

        # 获取68个关键点坐标与截取的人脸区域
        face_landmarks = puts.get_face_landmarks(frame, face_dlib)
        if not face_landmarks:
            puts.color_print(f"未检测到人脸关键点--> {frames_num}", "red")
            continue

        # 在frame上绘制人脸关键点
        puts.draw_face_landmarks(frame, face_landmarks, False)

        # 获取眼部矩形区域四角位置
        eye_location = puts.get_eye_location_PLUS(face_landmarks)

        # 在frame上绘制眼部矩形区域
        puts.draw_eye_location(frame, eye_location)

        # 获取左右眼的眼球中心坐标与眼球半径
        left_pupil_center, left_pupil_radius, right_pupil_center, right_pupil_radius\
            = puts.get_pupil_location(face_landmarks)

        # 在frame上绘制左右眼的眼球中心坐标与眼球半径
        puts.draw_pupil_location(frame, left_pupil_center, left_pupil_radius, (0, 0, 255))
        puts.draw_pupil_location(frame, right_pupil_center, right_pupil_radius, (0, 0, 255))

        # 获取眼部矩形区域图像
        eye_image = puts.get_eye_image(frame, eye_location)
        if not eye_image.any():
            puts.color_print(f"未检测到眼部区域--> {frames_num}", "red")
            continue

        # 放大眼部图像，使宽固定为400，高按比例缩放
        eye_w = 400
        eye_h = int(eye_image.shape[0] * eye_w / eye_image.shape[1])
        eye_image_enlarge = cv2.resize(eye_image, (eye_w, eye_h))

        # 显示图像
        cv2.imshow('frame', frame)
        cv2.imshow('eye_image', eye_image_enlarge)
        frames_num += 1

        if puts.wait_for_key(wait_time, 27):  # 按ESC退出
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()
