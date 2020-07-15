#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
# import keras.backend as K
import keras
from . import keys
from . import densenet


keras.backend.clear_session()
reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

input = Input(shape=(46, None, 1), name='the_input')
y_pred= densenet.dense_cnn(input, nclass)


basemodel = Model(inputs=input, outputs=y_pred)

#更改权重的目录
modelPath = os.path.join(os.getcwd(), os.getcwd() + r'/train/models/weights_densenet-09-0.11.h5')
#modelPath = "/home/zhang/CTPNOCR/SourceCode/train/models/weights_densenet-09-0.11.h5"
if os.path.exists(modelPath):
    g3 = tf.get_default_graph()
    with g3.as_default():
        basemodel.load_weights(modelPath)

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 46
    width = int(width / scale)
    
    img = img.resize([width, 46], Image.ANTIALIAS)
   
    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    
    X = img.reshape([1, 46, width, 1])
    
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    return out
