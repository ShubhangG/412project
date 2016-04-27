import pandas as pd
import numpy as np
from misc import *
from sklearn import cross_validation
import DTClassV3

from copy import copy
from time import clock
from collections import defaultdict
from operator import itemgetter

import csv


def reduceLabels(dataframe):
    del dataframe['id']
    del dataframe['date_account_created']
    del dataframe['date_first_booking']
    del dataframe['timestamp_first_active']

    return dataframe


def preprocessing(dataframe):
    dataframe = reduceLabels(dataframe)
    columns_to_encode = dataframe.columns.values.tolist()
    columns_to_encode = [x for x in columns_to_encode if not x.startswith('age')]
    encoder = Encoder(columns_to_encode)

    dataframe = encoder.encodeDataset(dataframe)
    labels = dataframe['country_destination']
    del dataframe['country_destination']
    numerical = np.array(dataframe)
    numerical = np.nan_to_num(numerical)
    return dataframe, numerical, labels, encoder


def createValidation(data, labels, test, train):
    sss = cross_validation.StratifiedShuffleSplit(labels, 1, test_size=test, train_size=train, random_state=0)
    for train_index, test_index in sss:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
    return X_train, X_test, y_train, y_test


def main():
    f = 'data/train_users_2.csv'
    dataframe = pd.DataFrame(pd.read_csv(f))
    """
    ['gender', 'age', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
    'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'country_destination']
    """
    dataframe, numerical, labels, encoder = preprocessing(dataframe)
    X_train, X_test, y_train, y_test = createValidation(numerical, labels, .3, .6)

    #decoded = encoder.decode(y_train, 'country_destination')
    #print decoded

    myTree = create_decision_tree(X_train, y_train.base)
    #DTClassV3.drawtree(myTree, jpeg='trerview.jpg')
    myList = []
    for row in range(0,100):#len(X_test)):
        y_test[row] = DTClassV3.classify(X_test[row],myTree)    #TODO sometimes invalid
        if isinstance(y_test[row], dict):
            myList.append(encoder.decode(int(round(y_test[row].keys()[0])),'country_destination'))
        else:
            myList.append(encoder.decode(y_test[row],'country_destination'))
        #print(y_test[row])

    thefile = open("outputFile.txt", 'w')
    for item in myList:
        thefile.write("%s\n" % item)
    #decoded = encoder.decode(y_test, 'country_destination')
    #print decoded

def create_decision_tree(xtrain, ytrain):
    # append y_train to x_train
    data = []
    print(len(ytrain))
    for row_index in range(0,2000):#len(ytrain)):
        #TODO should remove NDF?         or weight it in some way maybe?
        #if ytrain[row_index] == 7.0:
        #    continue

        data.append(np.append(xtrain[row_index], ytrain[row_index]))
        #print(xtrain[row_index])

    print(len(data))
    myTree = DTClassV3.buildtree(data)
    return myTree

if __name__ == '__main__':
    main()