#########################################################################################
####################### CREATIVE MECHANICAL ENGINEERING DESIGN(2) #######################
#######################               FIRST PROJECT               #######################
#######################          2016145063 Yang Hee Soo          #######################
#######################          2016145025 Lee Seong Won         #######################
#########################################################################################


# Import Essential Libraries
from keras import utils
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import  MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, glob, time
import cv2

# Modifying Learning Schedule during Trianing
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 45:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 35:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Build_Our Own Network
def BuildNet(width, height, depth, classes):

	# Initialize the model
	model = Sequential()
	inputShape = (height, width, depth)
 
	# if we are using "channels first", update the input shape
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)

	# Layer1
	model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	# Layer2
	model.add(Conv2D(64, (5, 5), padding="same"))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	# Layer3
	model.add(Conv2D(128, (5, 5), padding="same"))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	# Fully Connected Layer connected to RELU layers
	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
 
	# Softmax Classifier
	model.add(Dense(classes))
	model.add(BatchNormalization())
	model.add(Activation("softmax"))
 
	# Return Model
	return model

# Load Lego Images Function
def ImageLoad(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    return resized

# Load Training Lego Images Function 
def TrainLoad():
    LegoTrain_X = []
    LegoID_X = []
    LegoTrain_Y = []

    LegoFolders = [str(i) for i in range(16)]
    for f in LegoFolders:
        idx = LegoFolders.index(f)
        pth = os.path.join(test_path, f, '*g')
        files = glob.glob(pth)
        for fl in files:
            base = os.path.basename(fl)
            img = ImageLoad(fl)
            LegoTrain_X.append(img)
            LegoTrain_Y.append(idx)
            LegoID_X.append(base)
    print('Completed Loading Images')
    return LegoTrain_X, LegoTrain_Y, LegoID_X

# Load Testing Lego Images Function 
def TestLoad():
    path = os.path.join(test_path, 'test', '*g')
    files = sorted(glob.glob(path))

    LegoTest_X = []
    LegoID_Xtest = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = ImageLoad(fl)
        LegoTest_X.append(img)
        LegoID_Xtest.append(flbase)

    return LegoTest_X, LegoID_Xtest

# Normalize Loaded Lego Images
def NormalizeTrainData():
    LegoTrain_X, LegoTrain_Y, LegoID_X = TrainLoad()

    LegoTrain_X = np.array(LegoTrain_X, dtype=np.uint8)
    LegoTrain_Y = np.array(LegoTrain_Y, dtype=np.uint8)

    LegoTrain_X = LegoTrain_X.astype('float32')
    LegoTrain_X = LegoTrain_X / 255.
    LegoTrain_Y = np_utils.to_categorical(LegoTrain_Y, num_classes)

    return LegoTrain_X, LegoTrain_Y, LegoID_X

# Normalize Loaded Lego Test Images
def NormalizeTestData():
    LegoTest_X, LegoID_Xtest = TestLoad()

    LegoTest_X = np.array(LegoTest_X, dtype=np.uint8)
    LegoTest_X = LegoTest_X.astype('float32')
    LegoTest_X = LegoTest_X / 255

    print('Shape of testing data:', LegoTest_X.shape)
    return LegoTest_X, LegoID_Xtest

# Basic Information Setup Before Training
test_path = './Submit/Data/Train'
num_classes = 16
img_width, img_height = 200, 200
num_channels = 3
image_size = 200
nb_epoch = 50


# Data
LegoTrain_X, LegoTrain_Y, LegoID_X = NormalizeTrainData()
LegoTest_X, LegoID_Xtest = NormalizeTestData()
LegoTrain_X, LegoValid_X, LegoTrain_Y, LegoValid_Y = train_test_split(LegoTrain_X, LegoTrain_Y, test_size=0.2)

# Model
model = BuildNet(width=img_width, height=img_height, depth=num_channels, classes=num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callback
checkpoint = ModelCheckpoint("Legomodel.{epoch:03d}.hdf5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Model Fit
hist = model.fit(LegoTrain_X, LegoTrain_Y, batch_size=32, epochs=nb_epoch,callbacks=[checkpoint,lr_scheduler])

# Save the model
model.save_weights("./Submit/Results/weight.hdf5")

# Evaluate
results = model.evaluate(LegoValid_X, LegoValid_Y)
print('\n| Accuracy : ', results[1]*100,'%','\n| Loss : ' ,results[0])





