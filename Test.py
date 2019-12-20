import skimage as skid
import cv2
import pylab as plt
import scipy.misc
import os
import glob
import numpy as np
import sklearn as sk
import pickle
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from sklearn.svm import SVC

def GetDefaultParameters():
    image_size = (200,200)
    split = 0.2
    clusters = 500
    svm_c = 1
    kernel = 'linear'
    gamma = 'scale'
    step_size = 6
    bins = clusters
    parameters = {"imsize":image_size,"Split":split,"prepare":{"clusters":clusters,"step":step_size,"bins":bins},
              "train":{"svm_c":svm_c,"kernel":kernel,"gamma":gamma}}
    return parameters

def load_data(path,imsize):

    files = os.listdir(path)[0:3]
    labels = []
    img_list = []
    for file in files:
        newPath = path + "\\" + file
        for img in glob.glob(newPath + "\\*.jpg"):
            raw = cv2.imread(img, 0)
            im_data = np.asarray(raw)
            # gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
            sized = cv2.resize(im_data, imsize)
            img_list.append(sized)
            labels.append(file)
    img_array = np.array(img_list)
    data = {'Data':img_array,"Labels":labels}
    return data

def train_test_split(data, labels, ratio):
    #We need to change this into a simply 20 - 20 split, instead of the function
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(data, labels, test_size=ratio, random_state=42)
    split_dict = {"Train": {'Data': X_train, 'Labels': y_train}, "Test": {'Data': X_test, 'Labels': y_test}}
    return split_dict


def train_kmeans(data, params):
    # preparing the data
    # :Param
    # for every label and data computing sifts
    sift_vec = []
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create()
        step_size = params['step'] #using the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
        points, sifts = sift.compute(img, kp)
        sift_vec.append(sifts)

    all_sifts_array = np.array(sift_vec)

    model = MiniBatchKMeans(n_clusters=params["clusters"], random_state=42)
    kmeans = model.fit(all_sifts_array)

    return kmeans

def train(data, labels, params):

    svm = SVC(C=params['svm_c'], kernel=params['kernel'], gamma=params['gamma'])
    svm.fit(data, labels)
    return svm

def prepare(kmeans, data, labels, params):
    org_data = list(zip(labels, data))  # Creating a list of the labels and it suitable data
    label_vec = []
    hist_vec = []
    hist_dict = {}
    for label, img in org_data:
        sift = cv2.xfeatures2d.SIFT_create()
        step_size = params['step']
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
                range(0, img.shape[1], step_size)]
        points, sifts = sift.compute(img, kp)
        img_predicts = kmeans.predict(sifts)
        img_hist, bin_size = np.histogram(img_predicts, bins=params['bins'])
        hist_vec.append(img_hist)
        label_vec.append(label)
        img_hist = img_hist.reshape(1, 300)
        if label in hist_dict:
            hist_dict[label] = np.append(hist_dict[label], img_hist, axis=0)
        else:
            hist_dict[label] = img_hist
        datarep = {"hists":hist_vec,"labels":label_vec}
        return datarep

def test(model,test_data):
    predictions = model.predict(test_data)
    return predictions

def evaluate(predicts, real, params):
    accuracy = sk.metrics.accuracy_score(predicts, real)
    return accuracy

def reportResults(sum,params):
    print(sum)
    return



path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'

params= GetDefaultParameters()

DandL = load_data(path,params['imsize'])

SplitData= train_test_split( DandL['Data'], DandL['Labels'], params['Split'])
# returns train data, test data, train labels and test labels

kmeans_model = train_kmeans(SplitData['Train']['Data'], params['prepare'])
TrainDataRep = prepare( SplitData['Train']['Data'], params['Prepare'])
# call make_hist on train data
Model =  train(TrainDataRep, SplitData['Train']['Labels'] , params['Train'])
TestDataRep = prepare(SplitData['Test']['Data'], params['Prepare'])

Results = test(Model, TestDataRep)

Summary = evaluate(Results, SplitData['Test']['Labels'], params['Summary'])

reportResults(Summary, params['Report'])
