# 本文件实现了瞳孔追踪算法的实现，并提供了一些可视化的工具。
import cv2
import dlib
import numpy as np
import os
import ProUtils as puts
import threading

CAMERA_TYPE = 0  # 摄像头类型：0-笔记本内置摄像头，1-外接摄像头，2-网络摄像头
WAIT_TIME = 16  # 等待时间（秒）


def main(wait_time=WAIT_TIME, camera_type=CAMERA_TYPE):
    cap = cv2.VideoCapture(camera_type)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FPS, 30)   # 设置帧率为15

    while True:
        ret, frame = cap.read()  # 读取一帧图像
        if not ret:
            break

        # 获取计算的人脸位置
        face_location, face_dlib = puts.get_face_location(frame)
        if not face_location:
            puts.color_print("未检测到人脸")
            continue

        # 在frame上绘制人脸位置
        print("人脸位置：", face_location)
        puts.draw_face_location(frame, face_location)

        # 获取68个关键点坐标与截取的人脸区域
        face_landmarks = puts.get_face_landmarks(frame, face_dlib)
        if not face_landmarks:
            puts.color_print("未检测到人脸关键点")
            continue

        # 在frame上绘制人脸关键点
        puts.draw_face_landmarks(frame, face_landmarks)

        # 显示图像
        cv2.imshow('frame', frame)
        if puts.wait_for_key(wait_time, 27):  # 按ESC退出
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()
