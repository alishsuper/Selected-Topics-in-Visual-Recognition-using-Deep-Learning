# librarires
import os
import numpy as np
import cv2
import csv
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam

# train folder directory
DATADIR = "C:/Users/user/Desktop/PhD_108_fall/Selected Topics in Visual Recognition using Deep Learning/HW1/dataset1/dataset/train"

# test folder directory
DATADIR_TEST = "C:/Users/user/Desktop/PhD_108_fall/Selected Topics in Visual Recognition using Deep Learning/HW1/dataset/test"

# categories
CATEGORIES = {0: 'bedroom', 1: 'coast', 2: 'forest', 3: 'highway', 4: 'insidecity', 5: 'kitchen', 6: 'livingroom', 7: 'mountain', 8: 'office', 9: 'opencountry', 10: 'street', 11: 'suburb', 12: 'tallbuilding'}

# batch size
BatchSize = 64
Epochs = 40

# get test data
test_img_name = []
test_img = []
test_img_reshape = []

for dir_name, _, file_names in os.walk(DATADIR_TEST):
    for file_name in file_names:
        test_img_name.append(file_name)
        img_array = cv2.imread(os.path.join(dir_name, file_name))
        test_img.append(img_array)
    
for i in range(1040):
    new_array = cv2.resize(test_img[i], (300, 300), interpolation=cv2.INTER_LINEAR)
    test_img_reshape.append(new_array)

# data augmentation
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
generator = datagen.flow_from_directory(DATADIR, target_size=(300, 300), batch_size=BatchSize, class_mode='categorical', shuffle=True)

for i in range(1040):
    temp_reshape = preprocess_input(test_img_reshape[i])
    test_img_reshape[i] = temp_reshape.reshape(-1, 300, 300, 3)

train_step_size = generator.samples // generator.batch_size

# FIRST MODEL
vgg16_net1 = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
x1 = vgg16_net1.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dense(512, activation='relu')(x1)
x1 = Dropout(0.4)(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.4)(x1)
prediction1 = Dense(13, activation='softmax')(x1)

model1 = Model(inputs=vgg16_net1.input, outputs=prediction1)

# freeze the base VGG16 model
for layer in vgg16_net1.layers:
    layer.trainable = False

model1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model1.fit_generator(generator=generator, steps_per_epoch=train_step_size, epochs=Epochs)

# SECOND MODEL
vgg19_net1 = VGG19(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

x2 = vgg19_net1.output
x2 = GlobalAveragePooling2D()(x2)
x2 = Dense(512, activation='relu')(x2)
x2 = Dropout(0.4)(x2)
x2 = Dense(128, activation='relu')(x2)
x2 = Dropout(0.4)(x2)
prediction2 = Dense(13, activation='softmax')(x2)

model2 = Model(inputs=vgg19_net1.input, outputs=prediction2)

# freeze the base VGG19 model
for layer in vgg19_net1.layers:
    layer.trainable = False

model2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model2.fit_generator(generator=generator, steps_per_epoch=train_step_size, epochs=Epochs)

# THIRD MODEL
vgg16_net2 = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
x3 = vgg16_net2.output
x3 = GlobalAveragePooling2D()(x3)
x3 = Dense(512, activation='relu')(x3)
x3 = Dropout(0.4)(x3)
x3 = Dense(128, activation='relu')(x3)
x3 = Dropout(0.4)(x3)
prediction3 = Dense(13, activation='softmax')(x3)

model3 = Model(inputs=vgg16_net2.input, outputs=prediction3)

# freeze the base VGG16 model
for layer in vgg16_net2.layers:
    layer.trainable = False

model3.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model3.fit_generator(generator=generator, steps_per_epoch=train_step_size, epochs=Epochs)

# FOURTH MODEL
vgg19_net2 = VGG19(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

x4 = vgg19_net2.output
x4 = GlobalAveragePooling2D()(x4)
x4 = Dense(512, activation='relu')(x4)
x4 = Dropout(0.4)(x4)
x4 = Dense(128, activation='relu')(x4)
x4 = Dropout(0.4)(x4)
prediction4 = Dense(13, activation='softmax')(x4)

model4 = Model(inputs=vgg19_net2.input, outputs=prediction4)

# freeze the base VGG19 model
for layer in vgg19_net2.layers:
    layer.trainable = False

model4.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model4.fit_generator(generator=generator, steps_per_epoch=train_step_size, epochs=Epochs)

# FIFTH MODEL
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
x5 = resnet_model.output
x5 = GlobalAveragePooling2D()(x5)
x5 = Dense(2048, activation='relu')(x5)
x5 = Dropout(0.35)(x5)
x5 = Dense(128, activation='relu')(x5)
x5 = Dropout(0.35)(x5)
prediction5 = Dense(13, activation='softmax')(x5)

model5 = Model(inputs=resnet_model.input, outputs=prediction5)

# freeze the base ResNet50 model
for layer in resnet_model.layers:
    layer.trainable=False

model5.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model5.fit_generator(generator=generator, steps_per_epoch=train_step_size, epochs=Epochs)

# PREDICTION
with open('ensemble_model.csv', 'a') as csvFile:
    writer = csv.writer(csvFile, lineterminator='\n')
    writer.writerow(["id", "label"])
    
    for i in range(1040):
        ensemble_vote = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        name = test_img_name[i].split('.')[0]

        ensemble_vote[np.argmax(model1.predict(test_img_reshape[i]))] += 1
        ensemble_vote[np.argmax(model2.predict(test_img_reshape[i]))] += 1
        ensemble_vote[np.argmax(model3.predict(test_img_reshape[i]))] += 1
        ensemble_vote[np.argmax(model4.predict(test_img_reshape[i]))] += 1
        ensemble_vote[np.argmax(model5.predict(test_img_reshape[i]))] += 1
        writer.writerow([name, CATEGORIES[np.argmax(ensemble_vote)]])