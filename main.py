# from google.colab import drive
# drive.mount('/content/drive')
import chess
import cv2
import numpy as np
import tensorflow as tf
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import re
from cv_chess_functions import (read_img, canny_edge, hough_line, h_v_lines, line_intersections, cluster_points,
                                augment_points, write_crop_images, grab_cell_files, classify_cells, fen_to_image, atoi)

folder = 'Images'
image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rotation_range=5,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = datagen.flow_from_directory(
    folder + '/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    folder + '/test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

model = VGG16(weights='imagenet')
model.summary()

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Establish new fully connected block
x = base_model.output
x = Flatten()(x)  # flatten from convolution tensor output
x = Dense(500, activation='relu')(x)  # number of layers and units are hyperparameters, as usual
x = Dense(500, activation='relu')(x)
predictions = Dense(13, activation='softmax')(x)  # should match # of classes predicted

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

epochs = 10

history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    validation_data=test_gen
)
model.save_weights('model_VGG16.h5')

plt.plot(history.history['categorical_accuracy'], 'ko')
plt.plot(history.history['val_categorical_accuracy'], 'b')

plt.title('Accuracy vs Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

target_names = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR', 'Empty', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']

test_gen.reset()
Y_pred = model.predict_generator(test_gen)
classes = test_gen.classes[test_gen.index_array]
y_pred = np.argmax(Y_pred, axis=-1)
print(sum(y_pred == classes) / 800)

data = confusion_matrix(classes, y_pred)
df_cm = pd.DataFrame(data, columns=target_names, index=target_names)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font size

print('Confusion Matrix')
print(data)
print('Classification Report')
print(classification_report(test_gen.classes[test_gen.index_array], y_pred, target_names=target_names))


#################
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


img, gray_blur = read_img('Images/predict/alpha_data_image50.jpeg')
# Canny algorithm
edges = canny_edge(gray_blur)
# Hough Transform
lines = hough_line(edges)
# Separate the lines into vertical and horizontal lines
h_lines, v_lines = h_v_lines(lines)
# Find and cluster the intersecting
intersection_points = line_intersections(h_lines, v_lines)
points = cluster_points(intersection_points)
# Final coordinates of the board
points = augment_points(points)
# Crop the squares of the board a organize into a sorted list
x_list = write_crop_images(img, points, 0)
img_filename_list = grab_cell_files()
img_filename_list.sort(key=natural_keys)
# Classify each square and output the board in Forsyth-Edwards Notation (FEN)
fen = classify_cells(model, img_filename_list)
# Create and save the board image from the FEN
board = fen_to_image(fen)
# Display the board in ASCII
print(board)
# Display and save the board image
board_image = cv2.imread('current_board.png')
cv2.imshow('current board', board_image)
print('URL: https://www.chess.com/analysis?fen=' + fen)
print('Completed!')
