# Author: Ryan McCormick
# Slightly modified code modeled after: 
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

batch = 100
num_classes = 10
num_epochs = 45
data_augmentation = False

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Checking data sizes
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Feed-forward
model = Sequential()

"""Block 1"""
# Filters(32), Slider_size(5,5), input_shape(32,32,3)
model.add(Conv2D(32, (5, 5), strides=(1,1), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Activation('relu'))

"""Block 2"""
model.add(Conv2D(32, (5, 5), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2,2)))

"""Block 3"""
model.add(Conv2D(64, (5, 5), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))

"""Block 4"""
model.add(Conv2D(64, (4, 4), strides=(1,1), padding='same'))
model.add(Activation('relu'))

"""Block 5"""
model.add(Conv2D(10, (1, 1), strides=(1,1), padding='same'))
"""Optimizer"""
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False)

"""Loss Layer"""
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

if not data_augmentation:
	print('Not using data augmentation.')
	model.fit(x_train, y_train, batch_size=batch, epochs=num_epochs, 
				validation_data=(x_test, y_test), shuffle=True)
