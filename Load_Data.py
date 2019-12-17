import skimage as skid
import cv2
import pylab as plt
import scipy.misc
import os
import glob
import numpy as np

path = r'C:\Users\BIGVU\Desktop\Yoav\University\visionProject\101_ObjectCategories'
files = os.listdir(path)[0:3]

img_list = []
for file in files:
    newPath = path+"\\"+file
    for img in glob.glob(newPath+"\\*.jpg"):
        raw = cv2.imread(img)
        im_data = np.asarray(raw)
        gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        sized = cv2.resize(gray,(20,20))
        img_list.append(sized)
array = np.array(img_list)
print(array.shape)
print('opp')