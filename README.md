# 从零打造人脸识别系统
使用tensorflow,opencv等打造一个人脸识别系统。
本次从零开始打造一个人脸识别系统，主要分为两个部分,人脸检测与人脸识别。
首先快速的搭建一个能够跑起来的模型，然后逐步改进。
- 人脸探测
    - 人脸检测
    - 人脸对齐
    - 人脸剪裁
- 人脸验证
    - 基础模型的选择
    - 孪生网络的搭建

1. 首先解决人脸探测的问题。
	- 人脸检测
		- opencv级联分类器
			- 关于opencv级联分类器，它是使用haar特征的分类器
			- 首先使用了opencv的级联分类器，使用haarcascade_frontalface_alt2.xml进行人脸识别，
			发现其精度一般，经常会有误判的情况。随着现在检测的人脸越来越多变，室内到室外，有无眼镜，口罩遮挡，
			使得haar级联分类器表现不佳。
        - MTCNN
            - MTCNN是一种基于深度学习的人脸检测和人脸对齐方法
            - 原理
            - 结构，分为从粗到精的结构。
                - Proposal Network(P-Net)
                - Refine Network(R-Net)
                - Output Network(O-Net)
    - 人脸对齐
        - alpha0.1版本没有使用人脸对齐
    - 人脸剪裁
        - 根据人脸探测，与人脸对齐后的结果裁剪出人脸图片，作为人脸验证的输入。
2. 人脸验证
    - 人脸验证
        - 基础模型的选择
            - vgg16
                vgg16模型能够跑起来，但是模型非常庞大，速度较慢，基本上无法与摄像头使用达到即时识别
        - 孪生网络的搭建
            - 需要获得能够表达人脸特征的向量
            - 自定义损失函数
            - 自定义精度函数
            - 通过使用三元组进行训练
        - 加速验证速度
            - 将数据库中人物保存的图片改为保存人物特征向量。验证时直接使用向量进行验证。
            