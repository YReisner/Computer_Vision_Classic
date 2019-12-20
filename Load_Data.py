import skimage as skid
import cv2
import pylab as plt
import scipy.misc
import os
import glob
import numpy as np
import sklearn as sk
import pickle



#path = r'C:\Users\gilei\Desktop\comp\Computer_Vision_Classic-master\101_ObjectCategories'
files = os.listdir(path)[0:3]

labels = []
img_list = []
for file in files:
    newPath = path+"\\"+file
    for img in glob.glob(newPath+"\\*.jpg"):
        raw = cv2.imread(img, 0)
        im_data = np.asarray(raw)
        #gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        sized = cv2.resize(im_data,(200,200))
        img_list.append(sized)
        labels.append(file)
array = np.array(img_list)
""" For Testing
print(array.shape)
print(len(labels))
print(labels[50:60])
plt.imshow(array[400, :, :], cmap='gray')
plt.show()
"""



