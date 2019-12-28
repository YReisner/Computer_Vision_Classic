import skimage as skid
import cv2
import pylab as plt
import scipy.misc
import os
import glob
import numpy as np
import sklearn as sk
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from more_itertools import unique_everseen

def GetDefaultParameters():
    class_indices = [10,11,12,13,14,15,16,17,18,19]
    image_size = (150,150)
    split = 0.2
    clusters = 40
    svm_c = 200
    degree = 3
    kernel = 'linear'
    gamma = 5
    step_size = 6
    bins = clusters
    validate = False
    parameters = {"class_indices":class_indices,"validate":validate,"image_size":image_size,"Split":split,"clusters":clusters,"step_size":step_size,"bins":bins, "svm_c":svm_c,"kernel":kernel,"gamma":gamma,'degree':degree}
    return parameters


def load_data(path,params):
    '''
    Loads the data
    :param path: data location on the PC
    :param imsize: desired size measure for all images
    :return: a dictionary of images raw data and its labels
    '''

    files = os.listdir(path)  # get the files
    classes = [files[i] for i in params['class_indices']]  # Classes we choose for this run
    labels = []  # defining a list of labels
    img_list = []  # defining a list of images
    # running on all  files and all images in every file
    for file in classes:
        newPath = path + "\\" + file
        images = glob.glob(newPath + "\\*.jpg")
        if len(images) >= 40:
            amount = 40
        else:
            amount = len(images)
        for img in images[:amount]:
            raw = cv2.imread(img, 0)  # returns an an array of the image in gray scale
            im_data = np.asarray(raw)
            sized = cv2.resize(im_data, params["image_size"])  # resizing the data
            img_list.append(sized) # add it to list of the images
            labels.append(file)   # add the image label to th list of labels
    #img_array = np.array(img_list) # transfer the list to np.array
    #data = {'Data':img_array,"Labels":labels} # create a dict of images data and the labels
    print("Data loading complete!")
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
        step_size = params['step_size'] #use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
              range(0, img.shape[1], step_size)] # compute key points
        points, sifts = sift.compute(img, kp) #compute sifts from key points
        sift_vec.append(sifts) # add the sift to the sifts list

    # transfer the list to np.array
    all_sifts_array = list(sift_vec[0])
    for value in sift_vec[1:]:
        all_sifts_array = np.append(all_sifts_array, value, axis=0)
    # compute and return k_means
    model = MiniBatchKMeans(n_clusters=params["clusters"],  random_state=42,batch_size=params['clusters']*4)   # define the kmeans model parameters
    kmeans = model.fit(all_sifts_array) # fit the model on the sift and compute the kmeans
    print('Kmeans trained!')

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
        step_size = params['step_size'] #use the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
                range(0, img.shape[1], step_size)]  # compute key points
        points, sifts = sift.compute(img, kp) #compute sifts from key points
        img_predicts = kmeans.predict(sifts) # compute  k-means predictions for the computed sifts
        img_hist, bin_size = np.histogram(img_predicts, bins=params['bins']) #compute histogram for each image's sifts by 'bins' parameter
        normalized_hist = img_hist/sum(img_hist)
        hist_vec.append(normalized_hist) # add the histogram to histograms vector

    return hist_vec


def train(data, labels, params):
    '''
    Train the model with SVM
    :param data:
    :param labels:
    :param params:
    :return:
    '''
    svm = SVC(C=params['svm_c'], kernel=params['kernel'], gamma=params['gamma'],degree=params['degree'],probability = True,random_state=42) # define the SVM parameters
    svm.fit(data, labels)  # fitting the SVM on the data
    print('SVM Trained!')
    return svm


def test(model,test_data):
    '''

    :param model:
    :param test_data:
    :return:`q
    '''
    predictions = model.predict(test_data) #computing the predictions from the model
    probabilities = model.predict_proba(test_data)

    return predictions,probabilities


def evaluate(predicts, probabilities, real, params):

    error = 1 - sk.metrics.accuracy_score(predicts, real) # Compute error

    cnf_mat = confusion_matrix(real, predicts, list(unique_everseen(real)),normalize=True)  # Create confusion matrix

    # Create helper dict to access correct column in probabilities
    class_ind_dict = {real[0]:0}
    count = 1
    for i in list(range(1,len(real))):
        if real[i] != real[i-1]:
            class_ind_dict[real[i]] = count
            count += 1

    images_loc = []

    # Let's find the two worst pictures
    temporary_count = 3  # Help Variable

    if probabilities[0][class_ind_dict[real[0]]] > probabilities[1][class_ind_dict[real[1]]]:
        lowest = probabilities[1][class_ind_dict[real[1]]]
        second = probabilities[0][class_ind_dict[real[0]]]
        lowest_loc = 2
        second_loc = 1
    else:
        lowest = probabilities[0][class_ind_dict[real[0]]]
        second = probabilities[1][class_ind_dict[real[1]]]
        lowest_loc = 1
        second_loc = 2
    for i in list(range(2,len(real))):
        if real[i] == real[i-1]:
            if probabilities[i][class_ind_dict[real[i]]] < lowest:
                second = lowest
                second_loc = lowest_loc
                lowest = probabilities[i][class_ind_dict[real[i]]]
                lowest_loc = temporary_count

            elif probabilities[i][class_ind_dict[real[i]]] == lowest:
                continue

            elif probabilities[i][class_ind_dict[real[i]]] < second:
                second = probabilities[i][class_ind_dict[real[i]]]
                second_loc = temporary_count

            temporary_count += 1
        else:
            images_loc.append(lowest_loc+20)
            images_loc.append(second_loc+20)
            temporary_count = 3
            if probabilities[i][class_ind_dict[real[i]]] > probabilities[i+1][class_ind_dict[real[i+1]]]:
                lowest = probabilities[i+1][class_ind_dict[real[i+1]]]
                second = probabilities[i][class_ind_dict[real[i]]]
                lowest_loc = 2
                second_loc = 1
            else:
                lowest = probabilities[i][class_ind_dict[real[i]]]
                second = probabilities[i+1][class_ind_dict[real[i+1]]]
                lowest_loc = 1
                second_loc = 2
    images_loc.append(lowest_loc + 20)
    images_loc.append(second_loc + 20)
    print(images_loc)


    return error,cnf_mat


def reportResults(error,conf,labels):

    print(error)
    print(conf)
    return
'''
    df_cm = pd.DataFrame(conf, index=list(unique_everseen(labels)), columns=list(unique_everseen(labels)),)
    col_sum = df_cm.sum(axis=1)
    df_cm = df_cm.div(col_sum,axis = 0)
    plt.figure(figsize=(20,20))
    heatmap = sns.heatmap(df_cm, annot=True)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix, Normalized")
    plt.show()
'''


def validation(params,param_to_validate,possible_values):

    # Help variables to garner errors for each iteration
    train_errors = []
    val_errors = []

    for value in possible_values:
        params[param_to_validate] = value  # Change default value of tuned parameter to the one we want to check
        params['bins'] = value

        data, labels = load_data(path, params)
        SplitData = train_test_split(data, labels, params['Split'])
        # returns train data, test data, train labels and test labels

        kmeans_model = train_kmeans(SplitData['Train']['Data'], params)
        TrainDataRep = prepare(kmeans_model, SplitData['Train']['Data'], params)
        # call make_hist on train data

        Model = train(TrainDataRep, SplitData['Train']['Labels'], params)
        train_predicts = Model.predict(TrainDataRep) # Predict on the training set

        train_error = 1 - sk.metrics.accuracy_score(train_predicts, SplitData['Train']['Labels']) # Compute error
        print("current train error is %f" %(train_error))  # sanity check
        train_errors.append(train_error)  # Append training errors

        TestDataRep = prepare(kmeans_model, SplitData['Test']['Data'], params)

        Results, probabilities = test(Model, TestDataRep)

        Error, conf_mat = evaluate(Results,probabilities, SplitData['Test']['Labels'], [])
        reportResults(Error, conf_mat, [])

        val_errors.append(Error) # Append the current validation error for the current value

    # After checking all values - print the best error, value and for what parameter and plot a graph.
    print("The best error is %f, using the value %d, for parameter %s" %(min(val_errors),possible_values[val_errors.index(min(val_errors))],param_to_validate))

    plt.plot(possible_values,val_errors,label = 'Validation')
    plt.plot(possible_values, train_errors,label = 'Training')
    plt.xlabel(param_to_validate)
    plt.ylabel("Prediction Error")
    plt.title("Error as a function of %s" %(param_to_validate))
    plt.legend(loc='upper right')
    plt.show()
    return

np.random.seed(42)

path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'

params = GetDefaultParameters()

if not params['validate']:

    data,labels = load_data(path,params)

    SplitData = train_test_split(data, labels, params['Split'])
# returns train data, test data, train labels and test labels

    kmeans_model = train_kmeans(SplitData['Train']['Data'], params)
    TrainDataRep = prepare(kmeans_model, SplitData['Train']['Data'], params)
# call make_hist on train data
    Model = train(TrainDataRep, SplitData['Train']['Labels'], params)
    TestDataRep = prepare(kmeans_model, SplitData['Test']['Data'], params)

    Results,probabilities = test(Model, TestDataRep)

    Error,conf_mat = evaluate(Results,probabilities, SplitData['Test']['Labels'], [])

    reportResults(Error,conf_mat,SplitData['Test']['Labels'])

else:
    validation(params,'clusters',(40,80,100,150,200,250,300,350,400,450,500,550,600))
