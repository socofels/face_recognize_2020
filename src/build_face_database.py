from facedetect import facedetect
# 将人脸数据放到database中去
# 输入人脸，与人脸名称,保存人脸到人脸数据库
class build_face_database:

    # TODO(pl):直接从摄像头拍摄图片存入数据库的功能
    def camera(self):
        pass

    # todo 指定目录，将其中图片录入数据库
    def photos(self, pic_path):
        facedetect.predict().photos(pic_path,"face_database/")

build_face_database().photos("pic/database")