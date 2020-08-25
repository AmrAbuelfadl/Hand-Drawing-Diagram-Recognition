# Cropping the image 
import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread("/Users/amrabuelfadl/Desktop/difficult_image.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 10, 250)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
idx = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w>50 and h>50:
        idx+=1
        new_img=image[y:y+h,x:x+w]
        cv2.imwrite("output/"+str(idx) + '.png', new_img)

############################################################
# segmentation of image 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import os

def segmentation(file_path):
    img = cv2.imread(file_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    plt.imshow(gray, cmap='gray')
    ret, thresh = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 
    kernel = np.ones((3, 3), np.uint8) 
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                            kernel, iterations = 2) 
  # Background area using Dialation 
    bg = cv2.dilate(closing, kernel, iterations = 1) 
  
    # Finding foreground area 
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
    ret, fg = cv2.threshold(dist_transform, 0.02
                        * dist_transform.max(), 255, 0) 
    # fg.paste((255,255,255), [0,0,fg.size[0],fg.size[1]])


    dst = cv2.cornerHarris(fg,2,3,0.04)

#result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
# fg[dst>0.01*dst.max()]=[70]
    firstTime = True
    for i in range(fg.shape[0]):
        row = True
        white = False
        for j in range(fg.shape[1]-1):
            if( fg[i,j]==255 and firstTime ):
                firstTime = False
                i+=1
            if(fg[i,j]==255 and fg[i,j+1]==0 and white):
                white = False
            if(fg[i,j]==255 and fg[i,j+1]==0 and not(white) and row):
                white= True
                row = False
            if(white):
                fg[i,j]=255

# cv2.imshow('dst',fg)
    for i in range(fg.shape[0]):
        for j in range(fg.shape[1]-1):
            if(fg[i,j]==255):
                fg[i,j]=0
            else:
                fg[i,j]=255


    plt.imshow(fg, cmap='gray')
    plt.imsave(file_path,fg,cmap='gray')


directory = 'output/'

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        segmentation(directory+filename)

############################################################
# Prediction of image
import pandas as pd
import numpy as np
import cv2
import argparse

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading and compiling presaved trained CNN
model = load_model('drawing_classification.h5')

label = {0: "Circle", 1: "Square", 2: "Triangle"}
def predict_one(file_name):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (28, 28))
    img = np.reshape(img, [1, 28, 28, 3])
    classes = model.predict_classes(img)[0]
    category = label[classes]
#     print("\n {1} is the {0}".format(category, file_name))
    return category

directory = 'output/'
i=0
img =  image.copy()
for k in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[k,j]=255
        
for filename in os.listdir(directory):
    if filename.endswith(".png"):  
        c = cnts[i]
        maxInColumns = np.amax(c, axis=0)
        minInColumns = np.amin(c, axis=0)
        # coordinate in case of square
        xTopLeft = minInColumns[0,0]
        yTopLeft = maxInColumns[0,1]
        xBottomRight = maxInColumns[0,0]
        yBottomRight = minInColumns[0,1]
        # coordinate in case of Triange
        xFirstTri = minInColumns[0,0]
        yFirstTri = maxInColumns[0,1]
        xThirdTri = maxInColumns[0,0]
        yThirdTri = maxInColumns[0,1]
        xSecTri = (xFirstTri+xThirdTri)//2
        ySecTri = minInColumns[0,1]
        radius = (maxInColumns[0,0]-minInColumns[0,0])//2
        # coordinate in case of Circle
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"]) #circle
        cY = int(M["m01"] / M["m00"]) #circle
        if predict_one(directory+filename)=="Circle":
            cv2.circle(img, (cX,cY), radius, (0,255,0), 4)
        elif predict_one(directory+filename)=="Square":
            cv2.rectangle(img,(xTopLeft,yTopLeft),(xBottomRight,yBottomRight), (255,0,0), 3)
        else: 
            cv2.line(img, (xFirstTri,yFirstTri), (xSecTri,ySecTri), (0,0,255),5)
            cv2.line(img, (xSecTri,ySecTri), (xThirdTri,yThirdTri), (0,0,255),5)
            cv2.line(img, (xThirdTri,yThirdTri),(xFirstTri,yFirstTri), (0,0,255),5)
        i+=1
plt.imsave("output/output.jpg",img, cmap='gray')
