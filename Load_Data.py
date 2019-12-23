import skimage as skid
import cv2
import pylab as plt
import scipy.misc
import os
import glob
import numpy as np
import sklearn as sk
import pickle



path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'
files = os.listdir(path)[0:3]

labels = []  # defining a list of labels
img_list = []  # defining a list of images
# running on all  files and all images in every file
for file in files:
    newPath = path + "\\" + file
    images = glob.glob(newPath + "\\*.jpg")
    if len(images) >= 40:
        amount = 40
    else:
        amount = len(images)
    for img in images[:amount]:
        raw = cv2.imread(img, 0)  # returns an an array of the image in gray scale
        im_data = np.asarray(raw)
        sized = cv2.resize(im_data, (100,100))  # resizing the data
        img_list.append(sized) # add it to list of the images
        labels.append(file)   # add the image label
""" For Testing
print(array.shape)
print(len(labels))
print(labels[50:60])
plt.imshow(array[400, :, :], cmap='gray')
plt.show()
"""
print(len(img_list[1][1]))
plt.imshow(img_list[1], cmap='gray')
plt.show()


