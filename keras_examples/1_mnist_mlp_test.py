# multi-layer perceptron test

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# 1 data:
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
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
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# 3 training
batch_size = 128
epochs = 2
model.compile(loss='categorical_crossentropy',  # corrosponding to trick_2!
            optimizer=RMSprop(),
            metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test,y_test))
score = model.evaluate(x_test,y_test,verbose=0)                    
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

## transfer one-hot vector to number:
import numpy as np
y_result = np.argmax(y_predicted, axis=1)
print(y_result)

## input image display:
import matplotlib.pyplot as plt
plt.imshow(x_for_predict)
plt.show()

# 6 save & load model
## save:
## NOTE: You need to install h5py for this to work!
path_model = 'model_libn.h5'
model.save(path_model)
del model
## load:
from keras.models import load_model
model2 = load_model('model_libn.h5')
# predict
n = 12
x_for_predict = x_test[n,:]
x_for_predict = x_for_predict.reshape(1,784)
y_predicted2 = model2.predict(x_for_predict,verbose=1)
print(y_predicted2)
