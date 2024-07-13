import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret is None:
        break

    # 转化为灰度图
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 对灰度图进行滤波等处理，使其更加清晰便于人眼检测
    kernel = np.ones((5, 5), np.float32) / 25
    # 腐蚀
    gray_erode = cv2.filter2D(gray_frame, -1, kernel)
    # 膨胀
    gray_dilate = cv2.filter2D(gray_frame, -1, kernel)
    # 开运算
    gray_morphologyEX_open = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)
    # 闭运算
    gray_morphologyEX_close = cv2.morphologyEx(gray_frame, cv2.MORPH_CLOSE, kernel)
    # 边缘检测
    gray_canny = cv2.Canny(gray_frame, 100, 200)
    # 直方图均衡化
    gray_equalizeHist = cv2.equalizeHist(gray_frame)

    # 显示所有处理过的图像
    cv2.imshow('gray_frame', gray_frame)
    cv2.imshow('gray_erode', gray_erode)
    cv2.imshow('gray_dilate', gray_dilate)
    cv2.imshow('gray_morphologyEX_open', gray_morphologyEX_open)
    cv2.imshow('gray_morphologyEX_close', gray_morphologyEX_close)
    cv2.imshow('gray_canny', gray_canny)
    cv2.imshow('gray_equalizeHist', gray_equalizeHist)

    key = cv2.waitKey(25)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
