import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K  # 这里目的是使用后端tensorflow
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# 本图片认证采用的是三元组训练模型，构建孪生网络，将数据集分为一个一个的三元组进行训练
# 这里是全局训练参数调整

# 自定义损失函数
# 最大距离是1，最小距离是0
# 损失函数的公式是（1-Y）*0.5*（distance）^2 + y*0.5*max(0,margin-distance)^2
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # 这里需要指出的是，y_true表示的是两者是否为同一个人，1表示是同一人。
    # y_pred是距离，接近0表示是同一人。
    # 有了margin之后，margin-y_pred则转为与y_true相同。
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# 训练时的精度函数
def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    # 计算两个向量的距离。


def _euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # 向量距离 与 epsilon之间 取大的那个数，开根号，避免了等于0的情形。
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


# 表示了要输出的向量形状
def _eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def load_my_model(weightfile):
    model = verify()._model()
    model.load_weights(weightfile)
    return model


# 给定训练集的文件夹，返回两个数组，一个是成对图片的目录，一个是label
def _create_pic_pairs(pic_path, pairs_txt):
    f = open(pairs_txt)
    pairs_txt = f.readlines()[1:]
    pairs = []
    labels = []
    for temp in pairs_txt:
        temp = temp.split("\n")[0].split("\t")
        if len(temp) == 3:
            if len(temp[1]) != 4:
                temp[1] = (4 - len(temp[1])) * "0" + temp[1]
            if len(temp[2]) != 4:
                temp[2] = (4 - len(temp[2])) * "0" + temp[2]
            pair = [pic_path + "/" + temp[0] + "/" + temp[0] + "_" + temp[1] + ".jpg",
                    pic_path + "/" + temp[0] + "/" + temp[0] + "_" + temp[2] + ".jpg"]
            if os.path.exists(pair[0]) and os.path.exists(pair[1]):
                pairs.append(pair)
                labels.append([1])
        if len(temp) == 4:
            if len(temp[1]) != 4:
                temp[1] = (4 - len(temp[1])) * "0" + temp[1]
            if len(temp[3]) != 4:
                temp[3] = (4 - len(temp[3])) * "0" + temp[3]
            pair = [pic_path + "/" + temp[0] + "/" + temp[0] + "_" + temp[1] + ".jpg",
                    pic_path + "/" + temp[2] + "/" + temp[2] + "_" + temp[3] + ".jpg"]
            if os.path.exists(pair[0]) and os.path.exists(pair[1]):
                pairs.append(pair)
                labels.append([0])
    return np.array(pairs), np.array(labels)


class verify:

    def _load_image(self, a_pair_img_path):
        img1 = load_img(a_pair_img_path[0])
        img2 = load_img(a_pair_img_path[1])
        return img_to_array(img1) / 255.0, img_to_array(img2) / 255.0

    # 建立迭代器
    def _get_batch_img(self, x_samples, y_samples, batch_size):
        batch_num = int(len(x_samples) / batch_size)  # 有多少个batch
        max_len = batch_num * batch_size
        x_samples = x_samples[:max_len]  # 多余部分就不要了
        x_batches = np.split(x_samples, batch_num)  # 将x分割，x_batches每一个元素都是一个batch的目录
        y_samples = y_samples[:max_len]
        y_baches = np.split(y_samples, batch_num)
        while True:
            for i in range(batch_num):
                x = np.array(list(map(self._load_image, x_batches[i])))  # 输出每一个batch的图片
                y = np.array(y_baches[i]).astype("float32")
                yield [x[:, 0], x[:, 1]], y

    # 基础CNN网络,这里我使用的是VGG16结构,去掉原来softmax层，最后加了一层128向量。
    def _create_base_network(self):
        '''Base network to be shared (eq. to feature extraction).
        '''
        vgg16 = VGG16()
        base_model = Sequential()
        base_model.add(Model(inputs=vgg16.input, outputs=vgg16.get_layer("fc2").output, name="vgg16"))
        base_model.add(Dense(128, "relu", name="fc3"))
        # vgg层不进行训练权重
        base_model.layers[0].trainable = False
        return base_model

    # 最终用来计算精度函数
    def _compute_accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        # 返回的是布尔值pred
        # 最大
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)

    def _model(self, vgg_weight=None):
        input_shape = (224, 224, 3)
        model = self._create_base_network()
        # 开始构建孪生网络
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        processed_a = model(input_a)
        processed_b = model(input_b)

        # 增加一层，输入提取特征后的向量，输出两者距离
        distance = Lambda(_euclidean_distance,
                          output_shape=_eucl_dist_output_shape)([processed_a, processed_b])

        # 输入a,b,输出距离
        model = Model([input_a, input_b], distance)
        rms = RMSprop()
        model.compile(rms, loss=contrastive_loss, metrics=[accuracy])
        return model

    def train(self, face_path, batch_size, epochs, model_save_name="temp"):
        # 创建对应的数据集目录对
        pairs, label = _create_pic_pairs(face_path, "train_data/lfw/pairs.txt")
        # 设置训练集与测试集，测试集使用300个
        X_train_path, X_test_path, y_train, y_test = train_test_split(pairs, label, test_size=300, random_state=42)
        X_test = np.array(list(map(self._load_image, X_test_path)))
        model = self._model()
        model.fit_generator(self._get_batch_img(X_train_path, y_train, batch_size),
                            steps_per_epoch=len(y_train) // batch_size,
                            validation_data=([X_test[:, 0], X_test[:, 1]], y_test.astype("float32")),
                            epochs=epochs)
        pred = model.predict([X_test[:, 0], X_test[:, 1]])
        print(self._compute_accuracy(y_test, pred))
        model.save_weights(f"models/{model_save_name}.h5")

        return model.history

    def predict(self, pic_pairs, model):
        model = model
        pred = model.predict(pic_pairs)
        return pred
