import skimage as skid
import cv2
import pylab as plt
import scipy.misc
import os
import glob
import numpy as np
import sklearn as sk
import pickle
from sklearn.cluster import KMeans

def prepare(data, labels, param):
    org_data = list(zip(labels, data))
    sift_dict = {}
    for label, img in org_data:
        sift = cv2.xfeatures2d.SIFT_create()
        step_size = param['step']
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
        points, sifts = sift.compute(img, kp)
        if label in sift_dict:
            sift_dict[label].append(sifts)
        else:
            sift_dict[label] = [sifts]

        return sift_dict


path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'
#path = r'C:\Users\gilei\Desktop\comp\Computer_Vision_Classic-master\101_ObjectCategories'
files = os.listdir(path)[0:3]

labels = []
img_list = []
for file in files:
    newPath = path+"\\"+file
    for img in glob.glob(newPath+"\\*.jpg")[0:21]:
        raw = cv2.imread(img, 0)
        im_data = np.asarray(raw)
        #gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        sized = cv2.resize(im_data,(100,100))
        img_list.append(sized)
        labels.append(file)
array = np.array(img_list)


org_data = list(zip(labels, array))
sift_dict = {}
for label, img in org_data:
     sift = cv2.xfeatures2d.SIFT_create()
     step_size = 6
     kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
     points, sifts = sift.compute(img, kp)
     if label in sift_dict:
         sift_dict[label] = np.append(sift_dict[label],sifts,axis=0)
     else:
        sift_dict[label] = sifts

kmeans = KMeans(n_clusters=800,random_state=42)
bow = dict.fromkeys(sift_dict)


clusters = kmeans.fit(sift_dict["accordion"])
bow["accordion"] = clusters.cluster_centers_
print(bow['accordion'].shape)


