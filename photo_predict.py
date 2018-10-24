import numpy as np
import tensorflow as tf
from keras import backend as K
from tools import load_user_images
from keras.models import load_model
from tools import load_mnist_eval
from matplotlib import pyplot as plt
from tools import create_sub_dir


path_test_image = 'd:/test'

# 获得测试数据和名字
(data_images, data_names) = load_user_images(path_test_image)

# 改变维度为4D
data_images = np.array(data_images).reshape((data_images.shape[0], 28, 28, 1))

# 加载训练好的模型
model = load_model('./models/photos_model.h5')
model.load_weights('./models/photos_model_weights.h5')

# 开始预测分类
predict_result = model.predict_classes(data_images, batch_size=1, verbose=0)

# 显示预测到的分类结果
idx = 0
ret = False
for result in predict_result:
    print(result, '    |    %s' % (data_names[idx]))
    if result == data_names[idx]:
        ret = True
    else:
        ret = False
    print('-----------------', ret)
    idx += 1

# 清理内存
K.clear_session()
tf.reset_default_graph()
