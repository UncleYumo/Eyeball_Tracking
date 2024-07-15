import queue
import threading
import cv2
import dlib
import numpy as np
import os
import ProUtils as puts


def capture_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)


def process_frames(frame_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # 获取计算的人脸位置
        face_location = puts.get_face_location(frame)
        if not face_location:
            puts.color_print("未检测到人脸")
            continue

        print("人脸位置：", face_location)

        # 在frame上绘制人脸位置
        puts.draw_face_location(frame, face_location)

        cv2.imshow('frame', frame)

        if puts.wait_for_key(25, 27):  # 按ESC退出
            break


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_queue = queue.Queue()  # 定义队列

    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue,))

    capture_thread.start()
    process_thread.start()

    capture_thread.join()
    process_thread.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()