from utilis import *
from sklearn.model_selection import train_test_split

#Step 1 : Importing Data
path = 'AllPackHillTrack'
data = importDataInfo(path)

#Step 2 : Visulization and Distribution of Data
data = balanceData(data,display=True)

#Step 3 : Prepraing for Processing
imagesPath,steerings = loadData(path,data)
#print(imagesPath[0],steerings[0])
#print(imagesPathLeft[0],steerings[0])
#print(imagesPathRight[0],steerings[0])
#allImagesPath = list(zip(imagesPath,imagesPathLeft,imagesPathRight))

#Step 4 : Splitting Data (Training, Validation)
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)
print('Total Training Images :',len(xTrain))
#print('Total Training Images :',len(imagesPathLeft))
print('Total Validation Images :',len(xVal))
'''
trainImagesPath,_,_ = (zip(*xTrain))
_,trainImagesPathLeft,_ = (zip(*xTrain))
_,_,trainImagesPathRight = (zip(*xTrain))

valImagesPath,_,_ = (zip(*xVal))
_,valImagesPathLeft,_ = (zip(*xVal))
_,_,valImagesPathRight = (zip(*xVal))

print(trainImagesPath)
'''
#print(xVal)
#print('................................................................................................')
#print(xTrain)

#Step 5 : Augmentation of Data
#Step 6 : Preprocessing of Image
#Step 7 : Batch Generator

#Step 8 : Creating Model
model = createModel()
model.summary()

#Step 9 : Train Model

history = model.fit(batchGen(xTrain,yTrain,10,1),steps_per_epoch=500,epochs=20,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)
'''
history = model.fit(batchGen(trainImagesPath,trainImagesPathLeft,trainImagesPathRight,yTrain,10,1),steps_per_epoch=100,epochs=10,
          validation_data=batchGen(valImagesPath,valImagesPathLeft,valImagesPathRight,yVal,100,0),validation_steps=200)
'''
#Step 10 : Saving and Plotting Model
model.save('modelHillTrack3000spe_100e_v2_Right.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
#plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
