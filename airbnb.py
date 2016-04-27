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
    # del dataframe['country_destination']
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
    dataframe, numerical, labels, encoder = preprocessing(dataframe)
    X_train, X_test, y_train, y_test = createValidation(numerical, labels, .3, .6)
    # print encoder.decode(y_train, 'country_destination')
    print np.unique(y_train)
    print encoder.decode(np.unique(y_train), 'country_destination')
    k = KMeans(data=X_train, labels=y_train, k=12)
    result = k.findCenters()
    k.predict(X_test, y_test)


# k.initCentroids()

def create_decision_tree(xtrain, ytrain):
    """
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
    """
    data = xtrain  # now that we have re-appended to xtrain
    myTree = DTClassV3.buildtree(data)
    return myTree


def classify_on_DT(myTree, y_test, X_test, encoder):
    myList = []
    for row in range(0, 100):  # len(X_test)):
        y_test[row] = DTClassV3.classify(X_test[row], myTree)
        if isinstance(y_test[row], dict):  # TODO I have no idea why these are dicts
            myList.append(encoder.decode(int(round(y_test[row].keys()[0])), 'country_destination'))
        else:
            myList.append(encoder.decode(y_test[row], 'country_destination'))

    thefile = open("DToutputFile.txt", 'w')
    for item in myList:
        thefile.write("%s\n" % item)


if __name__ == '__main__':
    main()
