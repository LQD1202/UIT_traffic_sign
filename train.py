from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPool2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import ImageDataGenerator

raw_folder = "traffic_Data/DATA/"
def save_data(raw_folder=raw_folder):

    dest_size = (128, 128)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    # Lặp qua các folder con trong thư mục raw
    for folder in listdir(raw_folder):
        if folder!='.DS_Store':
            print("Folder=",folder)
            # Lặp qua các file trong từng thư mục chứa các em
            for file in listdir(raw_folder  + folder):
                if file!='.DS_Store':
                    print("File=", file)
                    pixels.append( cv2.resize(cv2.imread(raw_folder  + folder +"/" + file),dsize=(128,128)))
                    labels.append( folder)

    pixels = np.array(pixels)
    labels = np.array(labels)#.reshape(-1,1)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('pix.data', 'wb')
    # dump information to that file
    pickle.dump((pixels,labels), file)
    # close the file
    file.close()

    return

def load_data():
    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)


    return pixels, labels

save_data()
X,y = load_data()
#random.shuffle(X)
shuffle_indexes = np.arange(X.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = X[shuffle_indexes]
image_labels = y[shuffle_indexes]
X_train, X_test, y_train, y_test = train_test_split( image_data, image_labels, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

def get_model():
    # Tao model
    input = Input(shape=(128, 128, 3))
    conv11 = Conv2D(64, (3,3), (1,1), padding='same', activation='relu')(input)
    conv12 = Conv2D(64, (3,3), (1,1), padding='same', activation='relu')(conv11)
    conv12 = Dropout(0.5)(conv12)
    max1 = MaxPool2D((2,2))(conv12)

    conv21 = Conv2D(128, (3,3), (1,1), padding='same', activation='relu')(max1)
    conv22 = Conv2D(128, (3,3), (1,1), padding='same', activation='relu')(conv21)
    conv22 = Dropout(0.5)(conv22)
    max2 = MaxPool2D((2,2))(conv22)

    conv31 = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(max2)
    conv32 = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(conv31)
    conv33 = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(conv32)
    conv32 = Dropout(0.5)(conv32)
    max3 = MaxPool2D((2,2))(conv33)                                                       

    fl = Flatten()(max3)
    den1 = Dense(4096, activation='relu')(fl)
    den1 = Dropout(0.5)(den1)
    den2 = Dense(4096, activation='relu')(den1)
    den2 = Dropout(0.5)(den2)
    den3 = Dense(5, activation='softmax')(den2)

    # Compile
    my_model = Model(inputs=input, outputs=den3)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

from keras import models   
vggmodel = get_model()
#vggmodel = models.load_model('vggmodel.h5')
filepath="weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
    rescale=1./255,
	width_shift_range=0.1,
    height_shift_range=0.1,
	horizontal_flip=True,
    brightness_range=[0.2,1.5], fill_mode="nearest")

aug_val = ImageDataGenerator(rescale=1./255)

vgghist=vggmodel.fit_generator(aug.flow(X_train, y_train, batch_size=64),
                               epochs=100,# steps_per_epoch=len(X_train)//64,
                               validation_data=aug_val.flow(X_test,y_test,
                               batch_size=64),
                               callbacks=callbacks_list)

vggmodel.save("vggmodel.h5")
