import pandas as pd
import numpy as np
from misc import *
from sklearn import cross_validation
from copy import copy
from time import clock
from collections import defaultdict
from operator import itemgetter

def reduceLabels(dataframe):
	# del dataframe['id']
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
	labels = dataframe[['id', 'country_destination']]
	# print labels
	# del dataframe['country_destination']
	numerical = np.array(dataframe)
	# print numerical[0][0]
	labels = np.array(labels)
	# print labels[0][0]
	# print labels.ix[0][0]
	# print numerical[0][0] == labels[0][0]
	# print numerical[0][0] == labels.ix[0][0]

	# quit()
	numerical = np.nan_to_num(numerical)
	# print numerical[0][0] == labels.ix[0][0]
	# print labels[]
	return dataframe, numerical, labels, encoder

def createValidation(data, labels, test, train):
	# print labels
	# print labels[:,1]
	# quit()
	sss = cross_validation.StratifiedShuffleSplit(labels[:,1], 1, test_size=test, train_size=train, random_state=0)
	for train_index, test_index in sss:
		X_train, X_test = data[train_index], data[test_index]
		y_train, y_test = labels[train_index], labels[test_index]
	return X_train, X_test, y_train, y_test

def main():
	f = 'data/train_users_2.csv'
	dataframe = pd.DataFrame(pd.read_csv(f))
	dataframe, numerical, labels, encoder = preprocessing(dataframe)
	X_train, X_test, y_train, y_test = createValidation(numerical, labels, .4, .6)
	
	
	# print encoder.decode(y_train, 'country_destination')
	# print np.unique(y_train[:,1])
	# print encoder.decode(np.unique(y_train[:,1]), 'country_destination')
	k = KMeans(data=X_train, labels=y_train, k=12)
	result = k.findCenters()
	k.predict(X_test, y_test)
	f = 'data/test_users.csv'
	indf = pd.DataFrame(pd.read_csv(f))
	indf = reduceLabels(indf)
	indf = encoder.encodeDataset(indf)
	indf = np.array(indf)
	indf = np.nan_to_num(indf)
	predictions = k.predict(indf, test_labels=None, validate=True)
	print "id,country"

	countslist = list()
	for i in range(len(predictions)):

		try:
			for j in range(len(predictions[i])):
				entry = (encoder.decode(int(predictions[i][j][0]), 'id'), encoder.decode(i, 'country_destination'), len(predictions[i]))
				countslist.append(entry)

		except:
			continue
	countslist = sorted(countslist, key=itemgetter(2), reverse=True)
	for curr in countslist:
		print curr[0] + ',' + curr[1]
		# quit()
	# print len(predictions[4])
	# print predictions[0]

	# k.initCentroids()

if __name__ == '__main__':
	main()
