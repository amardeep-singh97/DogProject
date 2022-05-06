from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from PIL import ImageFile
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing import image                  
from tqdm import tqdm
from extract_bottleneck_features import *
import random
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline         

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets
    
# load train, test, and validation datasets
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("/data/lfw/*/*"))
print(human_files)
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
                      

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
plt.imshow(img)
plt.show()
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.
human_faces_1=[]
for imgs in human_files_short:
    human_faces_1.append(face_detector(imgs))
print("Percentage of Human Faces in Human's Pictures {} %".format((sum(human_faces_1)/len(human_faces_1))*100))

human_faces_2=[]
for imgs in dog_files_short:
    human_faces_2.append(face_detector(imgs))
print("Percentage of Human Faces in Dog's Pictures {} %".format((sum(human_faces_2)/len(human_faces_2))*100))

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

dogs_in_human=[]
for images in human_files_short:
    dogs_in_human.append(dog_detector(images))
print("Percentage of Dogs in Human's Pictures {} %".format((sum(dogs_in_human)/len(dogs_in_human))*100))

dogs_in_dogs=[]
for images in dog_files_short:
    dogs_in_dogs.append(dog_detector(images))
print("Percentage of Dogs in Dog's Pictures {} %".format((sum(dogs_in_dogs)/len(dogs_in_dogs))*100))

                          
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


model = Sequential()
model.add(Conv2D(filters= 16,kernel_size=(4,4),strides=2,padding='SAME',activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters= 32,kernel_size=(2,2),strides=1,padding='SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters= 64,kernel_size=(2,2),strides=1,padding='SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters= 64,kernel_size=(2,2),strides=1,padding='SAME',activation='relu'))
model.add(GlobalAveragePooling2D())

model.add(Dense(500,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(133,activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


epochs = 13

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

model.load_weights('saved_models/weights.best.from_scratch.hdf5')

# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

bottleneck_features = np.load('/data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


def VGG16_predict_breed(img_path):
   
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))

    predicted_vector = VGG16_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

bottleneck_features = np.load('/data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']


Resnet50_breed_model = Sequential()
Resnet50_breed_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_breed_model.add(Dense(300, activation='relu'))
Resnet50_breed_model.add(Dropout(0.4))
Resnet50_breed_model.add(Dense(150, activation='relu'))
Resnet50_breed_model.add(Dropout(0.2))
Resnet50_breed_model.add(Dense(133, activation='softmax'))

Resnet50_breed_model.summary()

Resnet50_breed_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)

Resnet50_breed_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=30, batch_size=20, callbacks=[checkpointer], verbose=1)

Resnet50_breed_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

Resnet50_breed_predictions = [np.argmax(Resnet50_breed_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

test_accuracy = 100*np.sum(np.array(Resnet50_breed_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

def predicted_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = Resnet50_breed_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

def detect_image(img_path):
    
    # if a face was detected or a dog was detected, return breed
    if face_detector(img_path):
        if dog_detector(img_path):
            return "Human with a dog of breed {}".format(predicted_breed(img_path))
        return "Hey Human! Predicted dog of breed {}".format(predicted_breed(img_path))
    if dog_detector(img_path):
        return "Dog of breed {}".format(predicted_breed(img_path))
    
    # return error otherwise
    return "This image may not contain dog or human"

#TESTING OUR ALGORITHM

img = cv2.imread('images/American_water_spaniel_00648.jpg')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/American_water_spaniel_00648.jpg')

img = cv2.imread('images/Curly-coated_retriever_03896.jpg')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/Curly-coated_retriever_03896.jpg')

img = cv2.imread('images/sample_cnn.png')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/sample_cnn.png')

img = cv2.imread('images/sample_dog_output.png')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/sample_dog_output.png')

img = cv2.imread('images/Welsh_springer_spaniel_08203.jpg')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/Welsh_springer_spaniel_08203.jpg')

img = cv2.imread('images/sample_human_output.png')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/sample_human_output.png')

img = cv2.imread('images/full_image.jpg')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/full_image.jpg')

img = cv2.imread('images/HD.jpeg')
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.show()
detect_image('images/HD.jpeg')