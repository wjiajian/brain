import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image

# 数据加载
(x_train,_),(x_test,_) = keras.datasets.mnist.load_data()

# 数据预处理
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 产生高斯分布的噪声
noise = np.random.normal(loc=0.5,scale=0.5,size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5,scale=0.5,size=x_test.shape)
x_test_noisy = x_test + noise

# 将像素值裁剪为[0.0，1.0]范围内
x_train_noisy = np.clip(x_train_noisy,0.0,1.0)
x_test_noisy = np.clip(x_test_noisy,0.0,1.0)

# 超参数
input_shape = (image_size,image_size,1)
batch_size = 32
kernel_size = 3
latent_dim = 16
layer_filters = [32,64]

"""
模型
"""
#编码器
inputs = keras.layers.Input(shape=input_shape,name='encoder_input')
x = inputs
for filters in layer_filters:
    x = keras.layers.Conv2D(filters=filters,
            kernel_size=kernel_size,
            strides=2,
            activation='relu',
            padding='same')(x)
shape = keras.backend.int_shape(x)

x = keras.layers.Flatten()(x)
latent = keras.layers.Dense(latent_dim,name='latent_vector')(x)
encoder = keras.Model(inputs,latent,name='encoder')
encoder.summary()

# 解码器
latent_inputs = keras.layers.Input(shape=(latent_dim,),name='decoder_input')
x = keras.layers.Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = keras.layers.Reshape((shape[1],shape[2],shape[3]))(x)
for filters in layer_filters[::-1]:
    x = keras.layers.Conv2DTranspose(filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            activation='relu')(x)
outputs = keras.layers.Conv2DTranspose(filters=1,
        kernel_size=kernel_size,
        padding='same',
        activation='sigmoid',
        name='decoder_output')(x)
decoder = keras.Model(latent_inputs,outputs,name='decoder')
decoder.summary

autoencoder = keras.Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()

# 模型编译与训练
autoencoder.compile(loss='mse',optimizer='adam')
autoencoder.fit(x_train_noisy,
        x_train,validation_data=(x_test_noisy,x_test),
        epochs=10,
        batch_size=batch_size)

# 模型测试
x_decoded = autoencoder.predict(x_test_noisy)

rows,cols = 3,9
num = rows * cols
imgs = np.concatenate([x_test[:num],x_test_noisy[:num],x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs,rows,axis=1))
imgs = imgs.reshape((rows * 3,-1,image_size,image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.imshow(imgs,interpolation='none',cmap='gray')
plt.show()

