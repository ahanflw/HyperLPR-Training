#coding=utf-8
import cv2
import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np


char_set = u"0123456789QWERTYUPASDFGHJKLZXCVBNM"


import os

f = file("config.txt","r")

T_folder = f.readline().strip()
F_folder = f.readline().strip()
CH_folder = f.readline().strip()

print T_folder,F_folder,CH_folder


list_T = []
list_F = []
list_CH = []
SIZE = 23



if __name__ == "__main__":

    for parent,dirnames,filenames in os.walk(T_folder):
            for filename in filenames:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    # print filename
                    list_T.append(os.path.join(parent,filename))

    for parent, dirnames, filenames in os.walk(F_folder):
            for filename in filenames:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    # print filename
                    list_F.append(os.path.join(parent,filename))

    for parent, dirnames, filenames in os.walk(CH_folder):
            for filename in filenames:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    list_CH.append(os.path.join(parent,filename))

    print(len(list_T),len(list_F),len(list_CH))
    if len(list_T) == 0 or len(list_CH)==0 or len(list_F)==0:
        raise "can't find files. please check your folder path."







#from keras import
def norm(image):
    return image.astype(np.float)/255

def pickone(list):
    name = list[np.random.randint(0,len(list)-1)]
    return name


from scipy.ndimage.filters import gaussian_filter1d

def Genernator(batchSize):
    while(1):
        # X = np.zeros(shape=(batchSize,23,23,1),dtype=np.float)
        # Y = np.zeros(shape =(batchSize,2),dtype=np.float)
        X = []
        Y = []

        for i in xrange(batchSize):
            # print pickone(list_T)
            r = np.random.random()
            if r>0.6:
                data =np.expand_dims(transfrom(cv2.resize((cv2.imread(pickone(list_T),cv2.IMREAD_GRAYSCALE)),(23,23))).astype(np.float)/255,3)
                X.append(data)
                Y.append(np.array([1.0,0.0,0.0],dtype=np.float))
            elif r>0.2:
                data =np.expand_dims(transfrom(cv2.resize((cv2.imread(pickone(list_F),cv2.IMREAD_GRAYSCALE)),(23,23))).astype(np.float)/255,3)
                X.append(data)
                Y.append(np.array([0.0,1.0,0.0],dtype=np.float))
            else:

                data =np.expand_dims(transfrom(cv2.resize((cv2.imread(pickone(list_CH),cv2.IMREAD_GRAYSCALE)),(23,23))).astype(np.float)/255,3)
                X.append(data)
                Y.append(np.array([0.0,0.0,1.0],dtype=np.float))

        # print "end"


        yield (np.array(X),np.array(Y))





from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')

import skimage.util

def transfrom( image):
    if np.random.random() > 0.5:
        image = cv2.equalizeHist(image)
    return image

def Getmodel_tensorflow_light(nb_classes):
    # nb_classes = len(charset)
    img_rows, img_cols = 23, 23
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters, nb_conv*2, nb_conv*2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(32))
    # model.add(Dropout(0.25))

    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def Getmodel_tensorflow_normal(nb_classes):
    # nb_classes = len(charset)
    img_rows, img_cols = 23, 23
    nb_filters = 16
    nb_pool = 2
    nb_conv = 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def TrainingWithGenerator():


    model = Getmodel_tensorflow_normal(3)
    # set = Genernate(100,char_set)
    BatchSize = 72*100
    # if os.path.exists("./char_judgement.h5"):
    #     model.load_weights("char_judgement.h5")

    model.fit_generator(generator=Genernator(128),samples_per_epoch=BatchSize,nb_epoch=30)
    model.save("char_judgement.h5")

if __name__ == "__main__":
    TrainingWithGenerator()




