import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import cv2
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam


def getName(filepath):
    return filepath.split('\\')[-1]


def importDataInfo(path):
    colums = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=colums)
    # print(data.head())
    # print(getName(data['Center'][0]))
    # print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    data['Left'] = data['Left'].apply(getName)
    data['Right'] = data['Right'].apply(getName)
    #print(data.head())
    print('Total Images Imported:', data.shape[0])
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 7000
    hist, bins = np.histogram(data['Steering'], nBins)
    #print(bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        #print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)

        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)

    print('Removed Images :',len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)
    print('Remaining Images :',len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        center = (bins[:-1] + bins[1:]) * 0.5
        #print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    return data

def loadData(path,data):
    imagesPath = []
    imagesPathLeft = []
    imagesPathRight = []
    steering = []

    for i in range(len(data)):
        indexdData = data.iloc[i]
        #print(indexdData)
        imagesPath.append(os.path.join(path,'IMG',indexdData[2]))
        #imagesPathLeft.append(os.path.join(path, 'IMG', indexdData[1]))
        #imagesPathRight.append(os.path.join(path, 'IMG', indexdData[2]))
        #print(os.path.join(path,'IMG',indexdData[0]))
        steering.append(float(indexdData[3]))

    imagesPath = np.asarray(imagesPath)
    #imagesPathLeft = np.asarray(imagesPathLeft)
    #imagesPathRight = np.asarray(imagesPathRight)
    steering = np.asarray(steering)

    return imagesPath,steering

def augmentImage(imgPath,steering):
    img = mpimg.imread(imgPath)
    #imgLeft = mpimg.imread(imgL)
    #imgRight = mpimg.imread(imgR)

    if np.random.rand()<0.5:
        #Translation
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.,),'y':(-0.1,0.1)})
        img = pan.augment_image(img)
        #imgLeft = pan.augment_image(imgLeft)
        #imgRight = pan.augment_image(imgRight)

    if np.random.rand() < 0.5:
        # Zoom
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
        #imgLeft = zoom.augment_image(imgLeft)
        #imgRight = zoom.augment_image(imgRight)
    if np.random.rand() < 0.5:
        # Brightness
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
        #imgLeft = brightness.augment_image(imgLeft)
        #imgRight = brightness.augment_image(imgRight)

    if np.random.rand() < 0.5:
        # Flip
        img = cv2.flip(img, 1)
        #imgLeft = cv2.flip(imgLeft, 1)
        #imgRight = cv2.flip(imgRight, 1)
        steering = -steering

    return img,steering

def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    '''
    imgLeft = imgLeft[60:135, :, :]
    imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_RGB2YUV)
    imgLeft = cv2.GaussianBlur(imgLeft, (3, 3), 0)
    imgLeft = cv2.resize(imgLeft, (200, 66))
    imgLeft = imgLeft / 255

    imgRight = imgRight[60:135, :, :]
    imgRight = cv2.cvtColor(imgRight, cv2.COLOR_RGB2YUV)
    imgRight = cv2.GaussianBlur(imgRight, (3, 3), 0)
    imgRight = cv2.resize(imgRight, (200, 66))
    imgRight = imgRight / 255
    '''
    return img

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        imgBatchLeft = []
        imgBatchRight = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img,steering = augmentImage(imagesPath[index],steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                imgLeft = mpimg.imread(imagesPath[index])
                imgRight = mpimg.imread(imagesPath[index])
                steering = steeringList[index]

            img = preProcessing(img)
            #imgLeft = preProcessing(imgLeft)
            #imgRight = preProcessing(imgRight)

            imgBatch.append(img)
            #imgBatchLeft.append(imgLeft)
            #imgBatchRight.append(imgRight)
            #allImage = list(zip(imgBatch,imgBatchLeft,imgBatchRight))
            steeringBatch.append(steering)

        yield (np.asarray(imgBatch),np.asarray(steeringBatch))
        #yield (np.asarray(allImage), np.asarray(steeringBatch))

def createModel():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2),  activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3),  activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss='mse')

    return model