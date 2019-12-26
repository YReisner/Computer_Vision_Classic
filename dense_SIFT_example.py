import skimage as skid
import cv2
import pylab as plt
import scipy.misc
import numpy as np
import os
import glob

path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'
file = os.listdir(path)[0]
newPath = path+"\\"+file
images = glob.glob(newPath+"\\*.jpg")

img = cv2.imread(images[6], 0)
#img = scipy.misc.face()
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sized = cv2.resize(img, (80,80))
gray = sized

plt.figure(figsize=(80,80))
plt.imshow(img,cmap='gray')
plt.show()

sift = cv2.xfeatures2d.SIFT_create()

step_size = 6
kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                                    for x in range(0, gray.shape[1], step_size)]

img=cv2.drawKeypoints(gray,kp, img)

plt.figure(figsize=(80,80))
plt.imshow(img,cmap='gray')
plt.show()

dense_feat = sift.compute(gray, kp)
print(dense_feat[1].shape)