import os

import cv2

cap = cv2.VideoCapture(0)
cap_fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率 cv2.CAP_PROP_FPS: 5
cap_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频尺寸
print('视频尺寸:', cap_size)
print('视频帧率:', cap_fps)

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目路径
video_save_path = os.path.join(project_path, 'out\\video.avi')  # 视频保存路径
print('项目路径:', project_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
print('视频编码:', fourcc)
out = cv2.VideoWriter(video_save_path, fourcc, cap_fps, cap_size)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)  # 写入视频，每帧保存一次
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        print("等待按键:" + str(key))
        if key == ord('q'):
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
