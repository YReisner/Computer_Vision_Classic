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

def GetDefaultParameters():
    image_size = (100,100)
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
    '''
    Loads the data
    :param path: data location on the PC
    :param imsize: desired size measure for all images
    :return: a dictionary of images raw data and its labels
    '''

    files = os.listdir(path)[0:2] #get the first 10 files into files
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



def prepare(kmeans, data, labels, params):
    '''

    :param kmeans:
    :param data:
    :param labels:
    :param params:
    :return:
    '''
    org_data = list(zip(labels, data))  # Create a list of the labels and the suitable data
    label_vec = [] # define a vector of labels
    hist_vec = [] # define a vector of histograms
    hist_dict = {} # define a dictionary of histograms
    for label, img in org_data: # run on the org_data tuple
        sift = cv2.xfeatures2d.SIFT_create() #############################
        step_size = params['step'] #use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
                range(0, img.shape[1], step_size)]  # compute key points
        points, sifts = sift.compute(img, kp) #compute sifts from key points
        img_predicts = kmeans.predict(sifts) # compute  k-means predictions for the computed sifts
        img_hist, bin_size = np.histogram(img_predicts, bins=params['bins']) #compute histogram for each image's sifts by 'bins' parameter
        hist_vec.append(img_hist) # add the histogram to histograms vector
        label_vec.append(label) # add the label to labels vector
        img_hist = img_hist.reshape(1, params['bins']) #reshape the size of each histogram of each image
        # fill the histogram dictionary with labels and historams
        if label in hist_dict:
            hist_dict[label] = np.append(hist_dict[label], img_hist, axis=0)
        else:
            hist_dict[label] = img_hist
    print(len(hist_vec))
    print(len(hist_vec[0]))
    datarep = {"hists":hist_vec,"labels":label_vec} #####################
    return datarep

def train(data, labels, params):
    '''
    Train the model with SVM
    :param data:
    :param labels:
    :param params:
    :return:
    '''
    svm = SVC(C=params['svm_c'], kernel=params['kernel'], gamma=params['gamma']) # define the SVM parameters
    svm.fit(data, labels)  # fitting the SVM on the data
    return svm

def test(model,test_data):
    '''

    :param model:
    :param test_data:
    :return:
    '''
    predictions = model.predict(test_data['hists']) #computing the predictions from the model
    return predictions

def evaluate(predicts, real, params):
    accuracy = sk.metrics.accuracy_score(predicts, real)
    return accuracy

def reportResults(sum,params):
    print(sum)
    return
def conf_matrix(true_lable, predicted_lable):
    '''

    :param true_lable:
    :param predicted_lable:
    :return:
    '''
    cnf_mat = confusion_matrix(true_lable, predicted_lable)
    return cnf_mat



path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'

params = GetDefaultParameters()

data,labels = load_data(path,params['imsize'])

SplitData = train_test_split(data, labels, params['Split'])
# returns train data, test data, train labels and test labels

kmeans_model = train_kmeans(SplitData['Train']['Data'], params['prepare'])
TrainDataRep = prepare(kmeans_model, SplitData['Train']['Data'],SplitData['Train']['Labels'], params['prepare'])
# call make_hist on train data
Model = train(TrainDataRep["hists"], SplitData['Train']['Labels'], params['train'])
TestDataRep = prepare(kmeans_model, SplitData['Test']['Data'], SplitData['Test']['Labels'], params['prepare'])

Results = test(Model, TestDataRep)

Summary = evaluate(Results, SplitData['Test']['Labels'], [])

reportResults(Summary,[])
