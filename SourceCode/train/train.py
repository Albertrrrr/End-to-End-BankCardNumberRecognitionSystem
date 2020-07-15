#-*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload
import densenet
import re


img_h = 46
img_w = 120
batch_size = 128
maxlabellength = 4

def get_session(gpu_fraction=1.0):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# def readfile(filename):
#     res = []
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         for i in lines:
#             res.append(i.strip())
#     dic = {}
#     regInt = "^0$|^[1-9]\d*$"
#     for i in res:
#         p = i.split(' ')
#         list_label = "".join(p[1:])
#         list_label = "".join(re.findall(regInt,list_label))
#         test = [str(index) for index in list_label]
#         dic[p[0]] = test
#
#     return dic

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]

    return dic

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str)

            if(len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) - 1 for k in str]
            labels[1,len(str):] = 10000

        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


if __name__ == '__main__':
    char_set = open('dict_str_10.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    nclass = len(char_set)
    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = '/home/zhang/CTPNOCR/SourceCode/train/models/weights_densenet-03-0.15.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    train_loader = gen('data_train.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen('data_test.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    #保存最佳模型
    checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5',
                                 monitor='val_loss', save_best_only=False, save_weights_only=True)
    '''
    keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=False, save_weights_only=False, mode='auto', period=1) 

参数说明：

filename：字符串，保存模型的路径
monitor：需要监视的值
verbose：信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
save_best_only：当设置为True时，监测值有改进时才会保存当前的模型（ the latest best model according to the quantity monitored will not be overwritten）
mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
period：CheckPoint之间的间隔的epoch数
---------------------

回调函数 Callbacks 是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
然后，在模型上调用 fit() 函数时，可以将 ModelCheckpoint 传递给训练过程。

训练深度学习模型时，Checkpoint 是模型的权重。ModelCheckpoint
回调类允许你定义检查模型权重的位置，文件应如何命名，以及在什么情况下创建模型的 Checkpoint。
---------------------


    '''
    # 动态设置学习率  learning_rate
    lr_schedule = lambda epoch: 0.0005 * 0.4**epoch
    learning_rate = np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))

    #防止过拟合 Early stopping（需要验证集） 早停法
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
    	steps_per_epoch =76800// batch_size,
    	epochs = 50,
    	initial_epoch = 0,
    	validation_data = test_loader,
    	validation_steps =512// batch_size,
    	callbacks = [checkpoint, earlystop, changelr, tensorboard])