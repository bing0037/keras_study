# Transfer-CNN for mnist test

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation, Flatten

import datetime
now = datetime.datetime.now

# 1 data
num_classes = 5
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data type:

# 1) create two datasets one with digits below 5 and one with 5 and above
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

# 2) add one dimension!
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_train_lt5 = x_train_lt5.reshape(x_train_lt5.shape[0], 1, img_rows, img_cols)
    x_test_lt5 = x_test_lt5.reshape(x_test_lt5.shape[0], 1, img_rows, img_cols)
    x_train_gte5 = x_train_gte5.reshape(x_train_gte5.shape[0], 1, img_rows, img_cols)
    x_test_gte5 = x_test_gte5.reshape(x_test_gte5.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train_lt5 = x_train_lt5.reshape(x_train_lt5.shape[0], img_rows, img_cols, 1)
    x_test_lt5 = x_test_lt5.reshape(x_test_lt5.shape[0], img_rows, img_cols, 1)
    x_train_gte5 = x_train_gte5.reshape(x_train_gte5.shape[0], img_rows, img_cols, 1)
    x_test_gte5 = x_test_gte5.reshape(x_test_gte5.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train_lt5 = x_train_lt5.astype('float32')
x_test_lt5 = x_test_lt5.astype('float32')
x_train_gte5 = x_train_gte5.astype('float32')
x_test_gte5 = x_test_gte5.astype('float32')

###### Trick_1
x_train_lt5 /= 255
x_test_lt5 /= 255
x_train_gte5 /= 255
x_test_gte5 /= 255

print('x_train_lt5.shape = ', x_train_lt5.shape)
print('x_test_lt5.shape = ', x_test_lt5.shape)
print('x_train_gte5.shape = ', x_train_gte5.shape)
print('x_test_gte5.shape = ', x_test_gte5.shape)

###### Trick_2
y_train_lt5 = keras.utils.to_categorical(y_train_lt5, num_classes)
y_test_lt5 = keras.utils.to_categorical(y_test_lt5, num_classes)
y_train_gte5 = keras.utils.to_categorical(y_train_gte5, num_classes)
y_test_gte5 = keras.utils.to_categorical(y_test_gte5, num_classes)

# 2 model
# number of convolutional filters to use
filters = 32
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size = 3
feature_layer = [
    Conv2D(filters, kernel_size,
            padding='valid',
            input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]
# create complete model
model = Sequential(feature_layer + classification_layers)
model.summary()

# 3 training for 5-digit classfication [0..4]:
batch_size = 128
epochs = 5
model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
t = now()                
model.fit(x_train_lt5, y_train_lt5,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_test_lt5, y_test_lt5))
model_feature = model            
print('Training time: %s' % (now() - t))            
score = model.evaluate(x_test_lt5, y_test_lt5, verbose=0)
print('Model for 0-4 classfication: Test loss:', score[0])
print('Model for 0-4 classfication: Test accuracy:', score[1])  

# 4 training for other 5-digit classfication [5..9]
## 1) model freezing:
for l in feature_layer:
    l.trainable = False
## 2) transfer training for dense layers for new classfication [5..9]
batch_size = 128
epochs = 5
model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
t = now() 
model.fit(x_train_gte5, y_train_gte5,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_test_gte5, y_test_gte5))
print('Training time: %s' % (now() - t)) 
score = model.evaluate(x_test_gte5, y_test_gte5, verbose=0)
print('Model for 5-9 classfication: Test loss:', score[0])
print('Model for 5-9 classfication: Test accuracy:', score[1]) 


# 5 predict & display
## 1) [0..4] classfication:
## predict:
n = 11
x_for_predict = x_test_lt5[n,:]
x_for_predict = x_for_predict.reshape(1,28,28,1)
y_predicted = model_feature.predict(x_for_predict,verbose=1)
## display:
x_for_predict = x_for_predict * 256
x_for_predict = x_for_predict.reshape(28,28)
x_for_predict = x_for_predict.astype('int')
print(x_for_predict)
print('Classfication result[0..4]:')
print(y_predicted)     

## 2) [5..9] classfication:
## predict:
n = 11
x_for_predict = x_test_gte5[n,:]
x_for_predict = x_for_predict.reshape(1,28,28,1)
y_predicted = model.predict(x_for_predict,verbose=1)
## display:
x_for_predict = x_for_predict * 256
x_for_predict = x_for_predict.reshape(28,28)
x_for_predict = x_for_predict.astype('int')
print(x_for_predict)
print('Classfication result[5..9]:')
print(y_predicted)      


# + display:
import matplotlib.pyplot as plt
plt.imshow(x_for_predict)
plt.show()