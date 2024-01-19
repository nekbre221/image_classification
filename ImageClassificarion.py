#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 07:47:37 2023

@author: nekui-tiefang
"""
# Important imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2 #pip install opencv-python
import random
from os import listdir
from sklearn.preprocessing import  LabelBinarizer
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, array_to_img
from keras.optimizers import Adam
from PIL import Image
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization#
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU
from sklearn.model_selection import train_test_split


# Listing directory

ls "Data/Intel Image Dataset/"

# Plotting 25 images to check dataset
plt.figure(figsize=(11,11))
path = "Data/Intel Image Dataset/mountain"
for i in range(1,26):
    plt.subplot(5,5,i)
    plt.tight_layout()
    rand_img = imread(path +'/'+ random.choice(sorted(listdir(path))))
    plt.imshow(rand_img)
    plt.title('mountain')
    plt.xlabel(rand_img.shape[1], fontsize = 10)
    plt.ylabel(rand_img.shape[0], fontsize = 10)
    
    
# Setting root directory path and creating empty list
dir = "Data/Intel Image Dataset/"
root_dir = listdir(dir)
image_list, label_list = [], []    


# Reading and converting image to numpy array
for directory in root_dir:
  for files in listdir(f"{dir}/{directory}"):
    image_path = f"{dir}/{directory}/{files}"
    image = Image.open(image_path)
    image = image.resize((150,150)) # All images does not have same dimension
    image = img_to_array(image)
    image_list.append(image)
    label_list.append(directory)
    
    
# Visualize the number of classes count
label_counts = pd.DataFrame(label_list).value_counts()
label_counts    

# Checking count of classes
num_classes = len(label_counts)
num_classes

# Checking x data shape
np.array(image_list).shape


# Checking y data shape
np.array(label_list).shape



# Splitting dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10) 


# Normalize and reshape data
x_train = np.array(x_train, dtype=np.float16) / 225.0
x_test = np.array(x_test, dtype=np.float16) / 225.0
x_train = x_train.reshape( -1, 150,150,3)
x_test = x_test.reshape( -1, 150,150,3)

"""
En résumé, cette séquence de code normalise les valeurs des pixels des images 
pour les ramener à une plage de 0 à 1, puis remodelle les tableaux d'images 
pour les adapter à un format 4D généralement utilisé dans les modèles 
d'apprentissage automatique convolutionnels.
"""


# Binarizing labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
print(lb.classes_)


# Splitting the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)



# Creating model architecture
model = Sequential([
        Conv2D(16, kernel_size=(3, 3), input_shape=(150, 150, 3)),
        BatchNormalization(),
        LeakyReLU(),
          
        Conv2D(32, kernel_size=(3, 3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling2D(5, 5),
        
        Conv2D(64, kernel_size=(3, 3)),
        BatchNormalization(),
        LeakyReLU(),
        
        Conv2D(128, kernel_size=(3, 3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling2D(5, 5),

        Flatten(),
    
        Dense(64),
        Dropout(rate=0.2),
        BatchNormalization(),
        LeakyReLU(),
        
        Dense(32),
        Dropout(rate=0.2),
        BatchNormalization(),
        LeakyReLU(),
    
        Dense(16),
        Dropout(rate=0.2),
        BatchNormalization(),
        LeakyReLU(1),
    
        Dense(6, activation='softmax')    
        ])
model.summary()


# Compiling model
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0005),metrics=['accuracy'])


# Training the model

history = model.fit(x_train, y_train, batch_size =128, epochs = 70, validation_data = (x_val, y_val))




# Saving model
model.save("Classification_intel_image.h5")


#Plot the loss history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'], color='b')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

# Calculating test accuracy 
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

# Storing model predictions
y_pred = model.predict(x_test)

# Plotting image to compare
img = array_to_img(x_test[1])
img

# Finding max value from predition list and comaparing original value vs predicted
labels = lb.classes_
print(labels)
print("Originally : ",labels[np.argmax(y_test[3])])
print("Predicted : ",labels[np.argmax(y_pred[3])])


















from keras.models import load_model

# Charger le modèle depuis le fichier
loaded_model = load_model("Classification_intel_image.h5")

# Maintenant, vous pouvez utiliser loaded_model pour faire des prédictions, par exemple :
# predictions = loaded_model.predict(x_test)

import matplotlib.pyplot as plt
from imageio import imread
from os import listdir
import os
plt.figure(figsize=(11, 11))
path = "sample_data"
image_files = sorted(listdir(path))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.tight_layout()
    img_path = os.path.join(path, image_files[i-1])
    rand_img = imread(img_path)
    plt.imshow(rand_img)
    plt.title('titre')
    plt.xlabel(rand_img.shape[1], fontsize=10)
    plt.ylabel(rand_img.shape[0], fontsize=10)
plt.show()

    
from os import listdir, path
from PIL import Image

# Répertoire contenant les images
dir = "sample_data"

# Liste pour stocker les images
images_list = []

# Liste de fichiers dans le répertoire
file_list = listdir(dir)

# Filtrer les fichiers pour ne conserver que les fichiers image (par exemple, avec une extension .jpg)
image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg'))]

# Charger chaque image dans la liste
for image_file in image_files:
    image_path = path.join(dir, image_file)
    
    # Utiliser PIL pour ouvrir l'image
    image = Image.open(image_path)
    image = image.resize((150,150)) # All images does not have same dimension
    image = img_to_array(image)
    # Ajouter l'image à la liste
    images_list.append(image)


# Normalize and reshape data
images_list = np.array(images_list, dtype=np.float16) / 225.0
images_list = images_list.reshape( -1, 150,150,3)


# Charger le modèle depuis le fichier
loaded_model = load_model("Classification_intel_image.h5")

# Maintenant, vous pouvez utiliser loaded_model pour faire des prédictions, par exemple :
# predictions = loaded_model.predict(x_test)
# buildings/  forest/  glacier/  mountain/  sea/  street/

pit=loaded_model.predict(images_list)
# Storing model predictions

# Plotting image to compare
array_to_img(images_list[0]) #building
array_to_img(images_list[1]) #mountain
array_to_img(images_list[2]) #street

pit=loaded_model.predict(images_list)
# Supposons que vous ayez des labels bien définis pour chaque colonne
labels = ['buildings','forest','glacier','mountain','sea','street']  # Exemple de labels, ajustez selon vos besoins


predicted_classes = np.argmax(pit, axis=1)

# Récupérer les labels associés aux indices prédits
predicted_labels = [labels[index] for index in predicted_classes]
