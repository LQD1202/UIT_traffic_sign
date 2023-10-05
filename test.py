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
import glob2
from keras.models import load_model
class_name = []
for i in range(0, 58):
    class_name.append(str(i))

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
    den1 = Dense(2048, activation='relu')(fl)
    den1 = Dropout(0.5)(den1)
    den2 = Dense(2048, activation='relu')(den1)
    den2 = Dropout(0.5)(den2)
    den3 = Dense(5, activation='softmax')(den2)

    # Compile
    my_model = Model(inputs=input, outputs=den3)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model
image_org_path = sorted(glob2.glob('traffic_Data/DATA/1/*.png'))[1]
image_org = plt.imread(image_org_path)
image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
# Resize
image = image_org.copy()
image = cv2.resize(image, dsize=(128, 128))
image = image.astype('float')*1./255
# Convert to tensor
image = np.expand_dims(image, axis=0)
# Load weights model da train
my_model = get_model()
#my_model.load_weights("weights-71-0.99.hdf5")
my_model = load_model('vggmodel.h5')

# Predict
predict = my_model.predict(image)
print("This picture is: ", class_name[np.argmax(predict[0])])
print(np.max(predict[0], axis=0))
print(predict)
