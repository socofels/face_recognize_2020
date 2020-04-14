import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from facedetect import facedetect
from verify import verify, load_my_model


# 人脸识别系统基本上分三步
# 1. 调用摄像头或者是读取图片
# 2. 人脸detect,检测图片中是否有人脸
# 3. 人脸验证,图片中人脸与数据库中人脸进行比对

# 前言,训练时对应的数据集每张图应该只有一张人脸。


class FaceRecognize:

    # 给一张图，识别图片中的人
    def photos(self, pic_paths,use_model):
        database_path = "face_database/"
        for pic_path in pic_paths:
            frame = cv.imread(pic_path)
            faces_loc = facedetect.predict().camere(frame)
            verify_model = load_my_model(f"models/{use_model}.h5")
            # 将每个脸与数据库中脸进行比对，更新pred_dis 与 pred_face
            for (x, y, w, h) in faces_loc:
                pred_dis = 1
                pred_face = "unknown"
                # 注意，opencv的坐标是以左上角为原点，向下和向右是正轴.
                up_left = (x, y)
                down_right = (x + w, y + h)
                face = frame[up_left[1]:down_right[1], up_left[0]:down_right[0]]
                # 将图片大小改为224*224
                face = cv.resize(face, (224, 224))
                face = face[np.newaxis]
                face = face/255
                for i in os.listdir(database_path):
                    # 依次读取数据库中每一张脸
                    database_face = cv.imread(database_path + i)
                    # 与数据库中连进行对比预测，pred是两个face特征向量的距离，越小代表越接近
                    database_face = database_face[np.newaxis]
                    database_face = database_face/255
                    pred = verify().predict([face, database_face], verify_model)
                    if pred < pred_dis:
                        pred_dis = pred
                        pred_face = i.split(".")[0]
                    print(f"与{i}特征距离为{pred}")
                # 在图上用框框标出这个人，并写下他的名字。
                frame = cv.rectangle(frame, up_left, down_right, (225, 225, 225))
                font = cv.FONT_HERSHEY_SIMPLEX
                high = (down_right[0] - up_left[0]) * 0.1 / 15
                if pred_dis < 0.5:
                    cv.putText(frame, pred_face, up_left, font, high, (255, 255, 255), 1, cv.LINE_AA)
                else:
                    cv.putText(frame, "unknown", up_left, font, high, (255, 255, 255), 1, cv.LINE_AA)
                cv.imshow(pred_face, frame)
                cv.waitKey()
                cv.destroyAllWindows()
    # 使用摄像头即时识别人
    def camera(self):
        pass

def plot_history(history):
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.plot(history["loss"])
    plt.legend(["train_acc", "test_acc", "loss"])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim([0,1])
    plt.show()


if __name__ == "__main__":
    # test_pic_path = ["pic/test/" + path for path in os.listdir("pic/test/")]
    # FaceRecognize().photos(test_pic_path,"v1")
    history = verify().train("train_data/lfw224",4,5,"v2")
    plot_history(history.history)
    pass