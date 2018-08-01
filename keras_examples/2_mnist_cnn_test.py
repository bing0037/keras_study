# CNN for mnist test

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

# 1 data
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data type:
# add one dimension!
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

###### Trick_1
x_train /= 255
x_test /= 255

print('x_train.shape = ', x_train.shape)
print('x_test.shape = ', x_test.shape)

###### Trick_2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 2 model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
        activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))        
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# 3 training
batch_size = 128
epochs = 12
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

model.fit(x_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test,y_test))                
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 4 predict
n = 11
x_for_predict = x_test[n,:]
x_for_predict = x_for_predict.reshape(1,784)
y_predicted = model.predict(x_for_predict,verbose=1)

# 5 display
x_for_predict = x_for_predict * 256
x_for_predict = x_for_predict.reshape(28,28)
x_for_predict = x_for_predict.astype('int')
print(x_for_predict)
print(y_predicted)
