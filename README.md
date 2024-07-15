### 基于 opencv-python 和 dlib 库的眼球追踪项目

> @author Uncle_Yumo
<br> @date 2024年7月
<br> @version 1.0.0
<br> @email <EMAIL>13921971147@163.com
---
**项目简介**
<br>
> 本项目基于 opencv-python 和 dlib 库实现了眼球追踪功能。通过摄像头捕捉视频流，检测人脸和眼部关键点，进而追踪瞳孔的位置和大小。项目还提供了一些可视化工具，帮助用户更好地理解和调试算法。
---
**功能特性**
- 时人脸检测
- 眼部关键点检测
- 瞳孔位置和大小追踪
- 眼部区域图像放大显示
- 可视化工具辅助调试
- 按键退出功能安装与使用
---
**安装依赖**
<br>
> 在开始使用本项目之前，请确保已安装以下依赖库：
```
pip install opencv-python
pip install dlib
```
本项目提供requirements.txt 文件，可使用 pip 命令安装依赖库,
但是由于pipreqs库的编码问题，包含了pip所有的依赖库，故不推荐使用：
```
pip install -r requirements.txt
```
本项目必须安装 dlib 库，根目录已经附带了预编译好的 dlib-19.19.0-cp38-cp38-win_amd64.whl 文件

使用方法：
```
pip cmake  # 安装 cmake 工具
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl  # 安装 dlib 库
```

---
**运行项目**
> 直接运行 main.py 文件即可启动眼球追踪功能：

配置参数
<br>
CAMERA_TYPE: 摄像头类型，默认为 0，表示使用默认摄像头。<br>
WAIT_TIME: 等待时间，单位为秒，默认为 16 秒。按 ESC 键可提前退出。

**代码结构**
```
├── main.py                # 主程序文件
├── src
│   ├── __init__.py        # 空文件，用于标识当前目录为 Python 包
    ├── Pupil_Tracking.py  # 眼球追踪实现
    └──ProUtils.py         # 工具函数库
├── data                   # 存放测试视频和图片
├── test.py                # 本人学习时的测试文件，仅供参考
├── requirements.txt       # 依赖库列表
├── dlib-19.19.0-cp38-cp38-win_amd64.whl.whl  # dlib 预编译包
└── README.md              # 项目说明文件
```
---
**主要函数说明**

main 函数<br>
打开摄像头并设置帧率，循环读取视频流，进行人脸和眼部关键点检测，
绘制人脸位置、关键点、眼部矩形区域和瞳孔位置，显示原始帧和放大后的眼部图像，
按 ESC 键退出程序

* ProUtils 工具库
* get_face_location: 获取人脸位置
* get_face_landmarks: 获取人脸关键点
* get_eye_location_PLUS: 获取眼部矩形区域
* get_pupil_location: 获取瞳孔中心和半径
* draw_face_location: 绘制人脸位置
* draw_face_landmarks: 绘制人脸关键点
* draw_eye_location: 绘制眼部矩形区域
* draw_pupil_location: 绘制瞳孔位置
* get_eye_image: 获取眼部区域图像
* color_print: 彩色打印输出
* wait_for_key: 等待按键输入
---
**注意事项！！！**

- 确保摄像头正常工作且无遮挡
- 环境光线对检测效果有一定影响，建议在光线适中的环境下使用
- 本项目python版本为3.8.10（64位），故项目附带的dilb预编译包为3.8版本，若您的python版本不同，请自行编译dlib
---

- **贡献**
欢迎提交 Issue 和 Pull Request，共同完善本项目。

---
**许可证**
本项目采用 MIT 许可证，详情请参见 LICENSE 文件。