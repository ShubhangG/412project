import pandas as pd
import numpy as np

import DTClassV3
from misc import *
from sklearn import cross_validation
from copy import copy
from time import clock
from collections import defaultdict
from operator import itemgetter


def reduceLabels(dataframe):
    del dataframe['date_account_created']
    del dataframe['date_first_booking']
    del dataframe['timestamp_first_active']
    #del dataframe['ids']

    return dataframe


def preprocessing(dataframe):

    ids = dataframe['id']
    dataframe = reduceLabels(dataframe)
    columns_to_encode = dataframe.columns.values.tolist()
    columns_to_encode = [x for x in columns_to_encode if not x.startswith('age')]
    encoder = Encoder(columns_to_encode)

    dataframe = encoder.encodeDataset(dataframe)
    labels = dataframe['country_destination']
    # del dataframe['country_destination']
    numerical = np.array(dataframe)
    numerical = np.nan_to_num(numerical)
    return dataframe, numerical, labels, encoder, ids


def createValidation(data, labels, test, train,ids):
    sss = cross_validation.StratifiedShuffleSplit(labels, 1, test_size=test, train_size=train, random_state=0)
    for train_index, test_index in sss:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        test_ids = ids[test_index]
    return X_train, X_test, y_train, y_test, test_ids


def main():
    f = 'data/train_users_2.csv'
    dataframe = pd.DataFrame(pd.read_csv(f))
    dataframe, numerical, labels, encoder, ids = preprocessing(dataframe)
    X_train, X_test, y_train, y_test, test_ids = createValidation(numerical, labels, .3, .6, ids)
    # print encoder.decode(y_train, 'country_destination')

    f_test ='data/test_users.csv'
    dataframe_real = pd.DataFrame(pd.read_csv(f_test))
    dataframe_real, numerical_real, labels_real, encoder_real, ids_real = reduceLabels(dataframe_real)
    X_train_real, X_test_real, y_train_real, y_test_real, test_ids_real = createValidation(numerical, labels, 1, 0, ids_real)


    tree = create_decision_tree(X_train)
    classify_on_DT(tree, y_test, X_test,encoder, test_ids)
    #This is what you need to call with the previously gnereated tree, the encoder, and the user ids
    #classify_real_DT(tree, ALL_DATA, encoder_real, test_ids_real)
"""
    print np.unique(y_train)
    print encoder.decode(np.unique(y_train), 'country_destination')
    k = KMeans(data=X_train, labels=y_train, k=12)
    result = k.findCenters()
    k.predict(X_test, y_test)
"""
# k.initCentroids()

def create_decision_tree(xtrain):

    print("creating tree")
    data = xtrain[0:2000]  # It takes too long to build the whole tree
    myTree = DTClassV3.buildtree(data)
    return myTree


def classify_on_DT(myTree, y_test, X_test, encoder,test_ids):
    correct = 0.0
    print("classifying with tree")
    thefile = open("outputFile.txt", 'w')
    myList = []
    y_test = np.array(y_test)
    ids = np.array(test_ids)

    for row in range(0, len(X_test)):
        prediction = DTClassV3.classify(X_test[row], myTree)
        if isinstance(prediction, dict):  # TODO I have no idea why these are dicts
            prediction = int(round(prediction.keys()[0]))
        if prediction == y_test[row]:
            correct += 1
        #myList.append(encoder.decode(prediction, 'country_destination'))
        thefile.write("%s\n" % ids[row]+ "," + (encoder.decode(prediction, 'country_destination')))

    print("My accuracy: " + str(correct / len(myList)))

#this is what you should call to create the output file
def classify_real_DT(myTree, X_test, encoder, test_ids):
    print("classifying with tree")
    thefile = open("outputFile.txt", 'w')
    ids = np.array(test_ids)
    for row in range(0, len(X_test)):
        prediction = DTClassV3.classify(X_test[row], myTree)
        thefile.write("%s\n" % ids[row]+ "," + (encoder.decode(prediction, 'country_destination')))

if __name__ == '__main__':
    main()
