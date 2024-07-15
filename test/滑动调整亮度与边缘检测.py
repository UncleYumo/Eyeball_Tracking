import cv2
import numpy as np


def CannyThreshold(lowThreshold):
    # 读取滑动条的值
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges,  # 边缘检测
                               lowThreshold,   # 低阈值
                               lowThreshold * ratio,   # 高阈值
                               apertureSize=kernel_size)  # 核大小
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # 显示效果
    cv2.imshow('canny demo', dst)


def TrackbarTest(current_value):
    # 读取滑动条的值
    print("当前值：", current_value)
    # 根据滑动条的值改变img的亮度
    img_bright = cv2.addWeighted(   # 改变亮度
        img,   # 原图
        current_value / 100.0,  # 亮度因子
        np.zeros(img.shape, img.dtype),  # 全黑色图
        0,
        0
    )
    cv2.imshow("Change Brightness", img_bright)


lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread("E:\\Dev_work\\Python_Dev\\Pycharm_Project\\eye_track_study\\data\images\\lxy01.jpg")
img = cv2.resize(img, (600, (img.shape[0] * 600 // img.shape[1])))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('canny demo')  # 创建一个名为“canny demo”的窗口

# # 在窗口中创建一个滑动条，用于调节阈值，并将其与函数CannyThreshold绑定，使其在滑动条变化时实时更新效果
# cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)

# # 调用函数CannyThreshold，显示效果
# CannyThreshold(0)  # initialization

cv2.namedWindow("Change Brightness")
cv2.createTrackbar("TrackbarTest", "Change Brightness", 0, 100, TrackbarTest)

if cv2.waitKey(0) == 27:
    print("程序已退出")
    cv2.destroyAllWindows()
