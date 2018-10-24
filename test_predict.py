import numpy as np
import tensorflow as tf
from keras import backend as K
from tools import load_user_images
from keras.models import load_model
from tools import load_mnist_eval
from matplotlib import pyplot as plt
from tools import create_sub_dir


def mnist_image_save():
    create_sub_dir('./mnist_image')
    nums = 0
    while nums < 10:
        num = 0
        path_save = './mnist_image/' + str(nums) + '/'
        count = 0

        while num < 1000:
            if data_names[num] == nums:
                filename = path_save + str(data_names[num]) + '_' + str(count) + '.png'
                plt.imsave(fname=filename, arr=data_images[num], cmap='Greys')
                count += 1
                if count == 100:
                    break
            num += 1
        nums += 1
    return


# 存放测试图片的路径
# path_test_image = './images'
path_test_image = 'mnist_image/0/'

# 获得测试数据和名字
(data_images, data_names) = load_user_images(path_test_image)
# (data_images, data_names) = load_mnist_eval('./data')


# mnist_image_save()
# exit(0)

# 改变维度为4D
data_images = np.array(data_images).reshape((data_images.shape[0], 28, 28, 1))

# 加载训练好的模型
model = load_model('./models/mnist_model.h5')
model.load_weights('./models/mnist_model_weights.h5')

# 开始预测分类
predict_result = model.predict_classes(data_images, batch_size=10, verbose=0)

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
