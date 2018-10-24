import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from tools import load_user_images
from matplotlib import  pyplot as plt
from keras.utils import plot_model


batch_size = 1                                    # 一次批处理100个样本
num_classes = 36                                    # 0-9数字一共10类
epochs = 100                                         # 对样本循环过10遍
img_rows = img_cols = 28                            # 样本的长
color_channels = 1                                  # 样本的颜色通道
input_shape = (img_rows, img_cols, color_channels)  # 样本的输入维度


# 加载数据集
(train_data, train_label) = load_user_images('./photos')
(eval_data, eval_label) = load_user_images('./photos')

# 重塑成4D
train_data = train_data.reshape(train_data.shape[0], img_rows, img_rows, color_channels)
eval_data = eval_data.reshape(eval_data.shape[0], img_rows, img_cols, color_channels)

# 转换类型
train_data = train_data.astype('float32')
eval_data = eval_data.astype('float32')

# 归一化
train_data /= 255
eval_data /= 255

# 将标签转换编码为ont-hot
train_label = keras.utils.to_categorical(train_label, num_classes)
eval_label = keras.utils.to_categorical(eval_label, num_classes)

# 搭建网络
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 打印模型信息
model.summary()

# 编译模型
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


# 训练模型
history = model.fit(x=train_data,
                    y=train_label,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(eval_data, eval_label))

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
fig.savefig('model_acc.png')
fig.clear()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
fig.savefig('model_loss.png')
fig.clear()


# 训练结果评估
score = model.evaluate(eval_data,
                       eval_label,
                       batch_size=batch_size,
                       verbose=1)

# 打印训练结果
print(model.metrics_names)
print('loss: ', score[0])
print('accuracy: ', score[1])

# 保存模型和权重
model.save('./models/photos_model.h5', overwrite=True)
model.save_weights('./models/photos_model_weights.h5', overwrite=True)

# 使用完模型之后，清空之前model占用的内存
K.clear_session()
tf.reset_default_graph()
