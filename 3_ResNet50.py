from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# model:
model = ResNet50(weights = 'imagenet')

# input:
img_path = 'rabit.jpeg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

# prediction:
preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])


# + Plot graph of model 
from keras.utils import plot_model
plot_model(model, to_file='ResNet_model.png')