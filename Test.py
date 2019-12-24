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
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def GetDefaultParameters():
    image_size = (100,100)
    split = 0.2
    clusters = 500
    svm_c = 1
    kernel = 'linear'
    gamma = 'scale'
    step_size = 6
    bins = clusters
    validate = True
    parameters = {"validate":validate,"imsize":image_size,"Split":split,"clusters":clusters,"step":step_size,"bins":bins, "svm_c":svm_c,"kernel":kernel,"gamma":gamma}
    return parameters


def load_data(path,imsize):
    '''
    Loads the data
    :param path: data location on the PC
    :param imsize: desired size measure for all images
    :return: a dictionary of images raw data and its labels
    '''

    files = os.listdir(path)[0:3] #get the first 10 files into files
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
            sized = cv2.resize(im_data, imsize)  # resizing the data
            img_list.append(sized) # add it to list of the images
            labels.append(file)   # add the image label to th list of labels
    #img_array = np.array(img_list) # transfer the list to np.array
    #data = {'Data':img_array,"Labels":labels} # create a dict of images data and the labels
    return img_list,labels


def train_test_split(data, labels, ratio):
    '''
    Splits the data and labels according to a ratio defined in Params.
    :param data:
    :param labels:
    :param ratio:
    :return: Dictionary of train (data and labels) and test (data and labels)
    '''
    #We need to change this into a simply 20 - 20 split, instead of the function
    unique_labels = set(labels)
    count = [(item,labels.count(item)) for item in unique_labels] # This gives me the amount of images for each category
    img_amount = dict(count)  # This allows a convenient access to the image count
    counter = 0  # Helper variable to go over the data vector
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # Let's go over the data and labels vectors as long as we are not out of bounds
    while counter < len(data):
        # First 20 images goes to training
        for num in range(20):
            X_train.append(data[counter])
            y_train.append(labels[counter])
            counter = counter + 1
        # Check if we have 40 images for this category
        if img_amount[labels[counter]] == 40:
            img_left = 20  # If we do, we should have exactly 20 images left to collect
        else:
            img_left = img_amount[labels[counter]]-20  # If we don't, we'll collect what is left for testing
        for num in range(img_left):
            X_test.append(data[counter])
            y_test.append(labels[counter])
            counter = counter + 1

    #X_train, X_test, y_train, y_test = sk.model_selection.train_tdest_split(data, labels, test_size=ratio, random_state=42)
    split_dict = {"Train": {'Data': X_train, 'Labels': y_train}, "Test": {'Data': X_test, 'Labels': y_test}}
    return split_dict


def train_kmeans(data, params):
    '''

    :param data:
    :param params:
    :return:
    '''
    sift_vec = []  # define a list of sift
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create() #############################
        step_size = params['step'] #use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
              range(0, img.shape[1], step_size)] # compute key points
        points, sifts = sift.compute(img, kp) #compute sifts from key points
        sift_vec.append(sifts) # add the sift to the sifts list

    # transfer the list to np.array
    all_sifts_array = list(sift_vec[0])
    for value in sift_vec[1:]:
        all_sifts_array = np.append(all_sifts_array, value, axis=0)
    # compute and return k_means
    model = KMeans(n_clusters=params["clusters"],  random_state=42)   # define the kmeans model parameters
    kmeans = model.fit(all_sifts_array) # fit the model on the sift and compute the kmeans

    return kmeans


def prepare(kmeans, data, params):
    '''

    :param kmeans:
    :param data:
    :param labels:
    :param params:
    :return:
    '''
    hist_vec = []  # define a vector of histograms
    for img in data:  # run on the org_data tuple
        sift = cv2.xfeatures2d.SIFT_create() #############################
        step_size = params['step'] #use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
                range(0, img.shape[1], step_size)]  # compute key points
        points, sifts = sift.compute(img, kp) #compute sifts from key points
        img_predicts = kmeans.predict(sifts) # compute  k-means predictions for the computed sifts
        img_hist, bin_size = np.histogram(img_predicts, bins=params['bins']) #compute histogram for each image's sifts by 'bins' parameter
        hist_vec.append(img_hist) # add the histogram to histograms vector

    return hist_vec


def train(data, labels, params):
    '''
    Train the model with SVM
    :param data:
    :param labels:
    :param params:
    :return:
    '''
    svm = SVC(C=params['svm_c'], kernel=params['kernel'], gamma=params['gamma'],probability = True,random_state=42) # define the SVM parameters
    svm.fit(data, labels)  # fitting the SVM on the data
    return svm


def test(model,test_data):
    '''

    :param model:
    :param test_data:
    :return:`q
    '''
    predictions = model.predict(test_data) #computing the predictions from the model
    probabilities = model.predict_proba(test_data)

    return predictions


def evaluate(predicts, real, params):

    error = 1 - sk.metrics.accuracy_score(predicts, real)

    cnf_mat = confusion_matrix(real, predicts)

    return error,cnf_mat


def reportResults(error,conf,params):

    print(error)
    print(conf)
    return

def validation(params,param_to_validate,possible_values):
    train_errors = []
    val_errors = []
    for value in possible_values:
        params[param_to_validate] = value
        data, labels = load_data(path, params['imsize'])

        SplitData = train_test_split(data, labels, params['Split'])
        # returns train data, test data, train labels and test labels

        kmeans_model = train_kmeans(SplitData['Train']['Data'], params)
        TrainDataRep = prepare(kmeans_model, SplitData['Train']['Data'], params)
        # call make_hist on train data

        Model = train(TrainDataRep, SplitData['Train']['Labels'], params)
        train_predicts = Model.predict(TrainDataRep)
        train_errors.append(1 - sk.metrics.accuracy_score(train_predicts, SplitData['Train']['Labels']))
        TestDataRep = prepare(kmeans_model, SplitData['Test']['Data'], params)

        Results = test(Model, TestDataRep)

        Error, conf_mat = evaluate(Results, SplitData['Test']['Labels'], [])
        reportResults(Error, conf_mat, [])
        val_errors.append(Error)
    print("The best error is %f, using the value %d, for parameter %s" %(min(val_errors),possible_values[val_errors.index(min(val_errors))],param_to_validate))
    plt.plot(possible_values,val_errors)
    plt.plot(possible_values, train_errors)
    plt.xlabel(param_to_validate)
    plt.ylabel("Prediction Error")
    plt.show()
    return

np.random.seed(42)

path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'

params = GetDefaultParameters()

if not params['validate']:

    data,labels = load_data(path,params['imsize'])

    SplitData = train_test_split(data, labels, params['Split'])
# returns train data, test data, train labels and test labels

    kmeans_model = train_kmeans(SplitData['Train']['Data'], params)
    TrainDataRep = prepare(kmeans_model, SplitData['Train']['Data'], params)
# call make_hist on train data
    Model = train(TrainDataRep, SplitData['Train']['Labels'], params)
    TestDataRep = prepare(kmeans_model, SplitData['Test']['Data'], params)

    Results = test(Model, TestDataRep)

    Error,conf_mat = evaluate(Results, SplitData['Test']['Labels'], [])

    reportResults(Error,conf_mat,[])

else:
    validation(params,'svm_c',(0.001,0.005,0.01,0.5,0.1,0.5,1,10))