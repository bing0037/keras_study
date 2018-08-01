# Complete project using keras:
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# 1 Data:
data = np.random.random((1000,1))
target = 100.0 * data + 5.0 + np.random.random((1000,1))

# 2 Model:
model = Sequential()
## first layer:
model.add(Dense(32, input_dim=1))
model.add(Activation('relu'))
## second layer:
model.add(Dense(1, activation='relu'))
## ... layer:
model.add(Dense(1))

# 3 Compilation(Training configuration):
# regression:
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# 4 Training:
model.fit(data, target, epochs=100, batch_size=20)

# # 5 Prediction:
# data_test = np.array([0.1,0.2,0.3])
# value = model.predict(data_test)
# print(value)

#################################
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)

#####################################


# + Plot graph of model 
from keras.utils import plot_model
plot_model(model, to_file='model.png')