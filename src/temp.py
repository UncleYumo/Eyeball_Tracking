import cv2
import numpy as np

cap = cv2.VideoCapture("eyes.mp4")

while (True):
    ret, frame = cap.read()
    if ret is False:
        break
    eye_frame = frame[100: 500, 157: 800]  # 利用切片工具，选出感兴趣eye_frame区域
    #  cv2.imshow("show",eye_frame)

    rows, cols, _ = eye_frame.shape  # 保存视频尺寸以备用
    gray_eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)  # 转灰度
    gray_eye_frame = cv2.GaussianBlur(gray_eye_frame, (7, 7), 0)  # 高斯滤波一次

    _, threshold = cv2.threshold(gray_eye_frame, 8, 255, cv2.THRESH_BINARY_INV)  # 二值化，依据需要改变阈值
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 画连通域
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # cv2.drawContours(eye_frame, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(eye_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(eye_frame, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
        cv2.line(eye_frame, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
        break

    cv2.imshow("eye_frame", eye_frame)
    cv2.imshow("Threshold", threshold)
    key = cv2.waitKey(30)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()