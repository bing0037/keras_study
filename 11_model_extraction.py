from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np

model = VGG19(weights='imagenet')
extracted_model = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)

image_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = extracted_model.predict(x)


# + Plot graph of model 
from keras.utils import plot_model
plot_model(model, to_file='full_model.png')
plot_model(extracted_model, to_file='extracted_model.png')