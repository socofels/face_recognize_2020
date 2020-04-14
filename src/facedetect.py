# 人脸识别系统基本上分三步
# 1. 调用摄像头，读取图片
# 2. 人脸检测
# 3. 人脸验证
import os
import time

import cv2 as cv

# 前言,训练时对应的数据集每张图应该只有一张人脸。
# 设置全局参数
# 剪裁出来的人脸大小
pic_size = (224, 224)


# 保存剪裁后的图片至指定目录
def _save_face(frame, up_left, down_right, path, filename):
    # now=time.strftime("%d__%H_%M_%S",time.localtime(time.time()))
    now = time.time()
    face = frame[up_left[1]:down_right[1], up_left[0]:down_right[0]]
    # resize图片,pic_size是全局参数
    face = cv.resize(face, pic_size)
    cv.imwrite(f"{path}/{filename}", face)


class facedetect():
    def __init__(self, pic_size):
        pass

    # TODO 这里是训练人脸检测模型，目前使用的是opencv的级联分类器，所以目前先空着
    def train(self, photos_path="train_data/lfw224"):
        pass

    # 返回所需要的东西
    class predict():
        # 返回图片的目录
        # todo 输入图片所在目录，图片保存目录，将检测到的人脸剪裁后保存。
        def photos(self, pic_path, pic_save_path,max_predict=None):
            # 读取图片的位置
            k = 0
            for pics in os.walk(pic_path):
                for pic in pics[2]:
                    dir = pics[0].split(pic_path,1)[1]
                    # pic_sizes是全局参数,剪裁后图片大小
                    frame_path = pic_path + dir + "/" + pic
                    frame = cv.imread(frame_path)

                    # 创建剪裁后的图片至detect_face
                    # ----------
                    face_cascade = cv.CascadeClassifier("opencv/haarcascade_frontalface_alt.xml")
                    # 转为灰度图，目的是加快检测速度
                    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
                    for (x, y, w, h) in faces:
                        if x:
                            # 如果文件夹不存在则创建对应的文件夹
                            save_dir = pic_save_path + dir
                            if os.path.exists(save_dir):
                                pass
                            else:
                                os.makedirs(save_dir + "/")
                            up_left = (x, y)
                            down_right = (x + w, y + h)
                            _save_face(frame, up_left, down_right, save_dir, pic)
                    # ----------
                # 达到设置的最大训练量时停止
                k += 1
                if k == max_predict:
                    break
            print(f"完成{k}张图片检测")

        # 立即返回图片中的每个人脸定位
        def camere(self, frame):
            face_cascade = cv.CascadeClassifier("opencv/haarcascade_frontalface_alt.xml")
            # 转为灰度图，目的是加快检测速度
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            # 检测人脸
            faces_loc = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            return faces_loc
