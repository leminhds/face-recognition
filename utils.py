import numpy as np
import os
import random

import cv2

from keras.preprocessing.image import load_img, img_to_array
from keras.backend import cast, square, sum, sqrt, maximum, mean, epsilon

def create_pairs(X, y, num_classes):
    pairs, labels = [], []# index of images in X and Y for each class
    class_idx = [np.where(y==i)[0] for i in range(num_classes)]
    # The minimum number of images across all classes
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1
  
    for c in range(num_classes):
        for n in range(min_images):
            # create positive pair
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[c][n+1]]
            pairs.append((img1, img2))
            labels.append(1)
      
            # create negative pair
            # first, create list of classes that are different from the current class
            neg_list = list(range(num_classes))
            neg_list.remove(c)
            # select a random class from the negative list. 
            # this class will be used to form the negative pair
            neg_c = random.sample(neg_list,1)[0]
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[neg_c][n]]
            pairs.append((img1,img2))
            labels.append(0)

    return np.array(pairs), np.array(labels)

def contrastive_loss(y_true, D):
    margin = 1
    
    return mean(np.float32(y_true) * square(D) + (1 - np.float32(y_true)) * maximum((margin-D),0))    

def euclidian_distance(vectors):
    vector1, vector2 = vectors
    sum_square = sum(square(vector1 - vector2), axis=1, keepdims=True)
    return sqrt(maximum(sum_square, epsilon()))
    

def load_data(DATA_DIR):
    X_train, y_train, X_test, y_test = [], [], [], []

    subfolders = sorted([file.path for file in os.scandir(DATA_DIR) if file.is_dir()])

    # iterate through the subfolders
    for idx, folder in enumerate(subfolders):
        for file in sorted(os.listdir(folder)):
            img = load_img(folder+ "/" + file, color_mode='grayscale')
            img = img_to_array(img).astype('float32')/255
            img = img.reshape(img.shape[0], img.shape[1],1)
            
            # get the first 35 as training data, and last 5 as testing
            # we label using idx: subject of folder 1 is assigned with label1
            if idx < 35:
                X_train.append(img)
                y_train.append(idx)
            else:
                X_test.append(img)
                y_test.append(idx-35)
                
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return (X_train, y_train), (X_test, y_test)

def write_on_frame(frame, text, text_x, text_y):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    box_coords = ((text_x, text_y), (text_x+text_width+20, text_y-text_height-20))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    return frame