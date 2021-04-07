import csv
import cv2
import numpy as np
import sklearn
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def load_file(file):    
    lines =[]
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines[1:]

# Take in an image from the center, left, or right camera of the car. 
# This is the input to your neural network.
def load_data(lines):
    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            correction = 0.2
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = '/opt/carnd_p3/data/IMG/' + filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line[3])
            if i == 1:
                measurement += correction
            elif i == 2:
                measurement -= correction
            measurements.append(measurement)
            
        # Flipping
        images.append(cv2.flip(image,1))
        measurements.append(-measurement)
        
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def generator(X_data,y_data, batch_size=32):
    num_samples = len(y_data)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            X_train = X_data[offset:offset+batch_size]
            y_train = y_data[offset:offset+batch_size]
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Each row in the below file correlates the image with the steering angle, throttle, brake, and speed of car. 
file = '/opt/carnd_p3/data/driving_log.csv'
lines = load_file(file)

train_s, valid_s = train_test_split(lines, shuffle=True, test_size=0.2)
X_train, y_train = load_data(train_s)
X_valid, y_valid = load_data(valid_s)

batch_size =32
train_set = generator(X_train,y_train,batch_size = batch_size)
valid_set = generator(X_valid,y_valid,batch_size = batch_size)

model = Sequential()

# Parallelize image normalization.
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(32,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(64,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(128,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(64,(3,3),activation = "relu"))
model.add(Conv2D(32,(3,3))) 
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10)) 
model.add(Dense(1))  

print(model.summary())
model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit_generator(train_set,
            steps_per_epoch=np.ceil(len(y_train)/batch_size), 
            validation_data=valid_set, 
            validation_steps=np.ceil(len(y_valid)/batch_size),
            epochs=5, verbose=1)

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('examples/image.png')

model.save('model.h5')
exit()
    