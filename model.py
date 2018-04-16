import csv

samples = []
with open('./mydata5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
from sklearn.utils import shuffle

correction = 0.3
batch_size = 32
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                for i in range(3):
                    source_path = batch_sample[i]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = "./mydata5/IMG/" + filename
                    image = cv2.imread(local_path)
                    images.append(image)
                    if i == 0: #center
                        images.append(np.fliplr(image))
                        images.append(augment_brightness_camera_images(image))
                    if i == 1: #left
                        iml, al_cor = random_transformation(image, angle)
                        images.append(iml)
                    if i == 2: #right
                        imr, ar_cor = random_transformation(image, angle)
                        images.append(imr)

                angles.append(angle)
                angles.append(-angle)
                angles.append(angle)
                angles.append(angle + correction)
                angles.append(al_cor)
                angles.append(angle - correction)
                angles.append(ar_cor)

            X_train = np.array(images)
            y_train = np.array(angles)
            #randomly ignore 80 % of data with zero steering angle
            X_train, y_train = remove_zero_steering(X_train, y_train, 0.8)
            yield shuffle(X_train, y_train)

# this code is from one of the blogs about the project.
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def remove_zero_steering(X, y, percentage):
    indices = [i for i, e in enumerate(y) if e == 0]
    count = int(len(indices) * percentage)
    if not count: return []  # edge case, no elements removed
    indices[-count:], removed = [], indices[-count:]

    X_nonzero = [i for j, i in enumerate(X) if j not in removed]
    y_nonzero = [i for j, i in enumerate(y) if j not in removed]
    X_ = np.asarray(X_nonzero)
    y_ = np.asarray(y_nonzero)
    return X_, y_

def random_transformation(image, angle):
    #rows, cols = image.shape
    x = np.random.uniform(0.0,0.5) * row
    acor = (x * 0.002) * angle
    y = np.random.uniform(0.0, 0.5) * col
    M = np.float32([[1, 0, x], [0, 1, y]])
    im = cv2.warpAffine(image, M, (col, row))
    return im , acor

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

from keras import backend as K
K.set_image_dim_ordering('tf')

ch, row, col = 3, 160, 320

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=7)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()