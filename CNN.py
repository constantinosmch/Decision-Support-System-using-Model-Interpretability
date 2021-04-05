# Import necessary libraries  
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D#, Maxpooling2D
import pickle
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from skimage.io import imread
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, RMSprop
import lime 
from lime import lime_image
from skimage.segmentation import mark_boundaries
# Specifies model location
filename = './models/CNN.sav'
# Declares the static categories from the dataset
CATEGORIES = ['no','yes']
# Reads the dataset
DATADIR = './Data/MRI_dataset/'
IMG_SIZE = 128
training_data=[]
# Creates training data 
def create_training_data():
        for category in CATEGORIES:
                path = os.path.join(DATADIR, category)
                class_num = CATEGORIES.index(category)
                for img in os.listdir(path):
                        try:
                                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
                                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                                training_data.append([new_array,class_num])
                        except Exception as e:
                                pass
create_training_data()
# Declares X and y arrays
X = []
y = []
# Assigns X as the image array and y as the labels
for features, label in training_data:
        y.append(label)
        X.append(features)
Z = np.array(X)
# Reshapes array
Z= Z.reshape(len(X),IMG_SIZE,IMG_SIZE,3)
X = Z
# Normalises data
X = X/255.0
# Splits data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=53)
y_train = to_categorical(y_train)
y_test =  to_categorical(y_test)
# Uncomment to re-train model
#adds all the layers
# model = Sequential()
# model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,3)))
# model.add(Conv2D(32, kernel_size=3, activation="relu"))
# #model.add(Maxpooling2D(pool_size= (2,2)))
# model.add(Flatten())
# model.add(Dense(len(CATEGORIES), activation="softmax"))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #trains model
# model.fit(X_train, y_train, epochs=3, batch_size= 1)
# pickle.dump(model, open(filename, 'wb'))

# Loads pre-trained model
inet_model = pickle.load(open(filename, 'rb'))

# Transforms image into appropriate size in order of the explainer to proccess the data
def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

# Uses image y1 as an example and transforms the image    
images = transform_img_fn([os.path.join('./Data/MRI_dataset/yes','y1.jpg')])

# Generates raw prediction
prediction = inet_model.predict(images)[0]

print("Raw prediction made by model:", prediction)

# Gets the element with highest confidence
most_conf_index = np.argmax(prediction)
answer_confidence = prediction[most_conf_index]

print("Model classified image as", CATEGORIES[most_conf_index], "with", answer_confidence,"confidence")

# Reshapes image
images = images.reshape(IMG_SIZE,IMG_SIZE,3)

# Produces explanation using the CNN model and the image instace
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images, inet_model.predict, top_labels=5, hide_color=1, num_samples=1000)

# Uncomment this to view the various different ways the explanation can be presented
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
# plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=False)
# plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

# plots the explanation
plt.show()

# Uncomment this section to see a no turmour detected example
# images = transform_img_fn([os.path.join('./Data/MRI_dataset/no','n11.jpg')])
# prediction = inet_model.predict(images)[0]
# print("Raw prediction made by model:", prediction)
# #Get the element with highest confidence
# most_conf_index = np.argmax(prediction)
# answer_confidence = prediction[most_conf_index]
# print("Model classified image as", CATEGORIES[most_conf_index], "with", answer_confidence,"confidence")