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

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, glob, time
import cv2

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

# Normalize Loaded Lego Images
def NormalizeTrainData():
    LegoTrain_X, LegoTrain_Y, LegoID_X = TrainLoad()

    LegoTrain_X = np.array(LegoTrain_X, dtype=np.uint8)
    LegoTrain_Y = np.array(LegoTrain_Y, dtype=np.uint8)

    LegoTrain_X = LegoTrain_X.astype('float32')
    LegoTrain_X = LegoTrain_X / 255.
    LegoTrain_Y = np_utils.to_categorical(LegoTrain_Y, num_classes)

    return LegoTrain_X, LegoTrain_Y, LegoID_X

# Basic Information Setup Before Training
test_path = os.path.dirname(os.path.realpath(__file__))+'/Data/Test'
num_classes = 16
img_width, img_height = 200, 200
num_channels = 3
image_size = 200

# Generate Lego Test Data
LegoTest_X, LegoTest_Y, LegoID_X = NormalizeTrainData()

# Build Network
model = BuildNet(width=img_width, height=img_height, depth=num_channels, classes=num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load Network
weight_dir = os.path.dirname(os.path.realpath(__file__))+'/Results/weight.hdf5'
model.load_weights(weight_dir)


# Evaluate Network
results = model.evaluate(LegoTest_X, LegoTest_Y)
print('\n| Accuracy : ', results[1]*100,'%','\n| Loss : ' ,results[0])







