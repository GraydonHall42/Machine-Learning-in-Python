# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

training_set_path = r'C:\Users\grayd\Downloads\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\training_set'
test_set_path = r'C:\Users\grayd\Downloads\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set'

# create train_datagen to augment our images 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# apply train_datagen object to our dataset. 
training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image

single_img_path = r'C:\Users\grayd\Downloads\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\single_prediction'

# create image object, and resize to 64x64
# kept in downloads to keep out of onedrive!
test_image = image.load_img(single_img_path+'/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)  # convert image to 2D array
test_image = np.expand_dims(test_image, axis = 0)  # have to add a 3rd dimension for the batch
result = cnn.predict(test_image)  # make prediction for test image
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)