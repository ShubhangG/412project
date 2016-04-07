import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import preprocessing
from copy import copy
from time import clock
from collections import defaultdict
from operator import itemgetter
pd.set_option('max_columns', 50)
current = clock()
f = 'data/train_users_2.csv'
curr = pd.read_csv(f)

df2 = pd.DataFrame(curr)

del curr
del df2['id']
del df2['date_account_created']
del df2['date_first_booking']
del df2['timestamp_first_active']
labels = df2['country_destination']

del df2['country_destination']
encoders = list()
encoders.append((preprocessing.LabelEncoder(), 'gender'))
encoders.append((preprocessing.LabelEncoder(), 'signup_method'))
encoders.append((preprocessing.LabelEncoder() , 'signup_flow'))
encoders.append((preprocessing.LabelEncoder(), 'language'))
encoders.append((preprocessing.LabelEncoder(), 'affiliate_channel'))
encoders.append((preprocessing.LabelEncoder(), 'affiliate_provider'))
encoders.append((preprocessing.LabelEncoder(), 'first_affiliate_tracked'))
encoders.append((preprocessing.LabelEncoder(), 'signup_app'))
encoders.append((preprocessing.LabelEncoder(), 'first_device_type'))
encoders.append((preprocessing.LabelEncoder(), 'first_browser'))

for encoder in encoders:
	df2[encoder[1]] = encoder[0].fit_transform(df2[encoder[1]])

df2 = np.array(df2)

df2 = np.nan_to_num(df2)
sss = cross_validation.StratifiedShuffleSplit(labels, 1, test_size=.3, train_size=.6, random_state=0)
for train_index, test_index in sss:
	X_train_shuffled, X_test_shuffled = df2[train_index], df2[test_index]
	y_train_shuffled, y_test_shuffled = labels[train_index], labels[test_index]


forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train_shuffled, y_train_shuffled)

infile = pd.read_csv('data/test_users.csv')
indf = pd.DataFrame(infile)
ids = indf['id']
del indf['id']
del indf['date_account_created']
del indf['date_first_booking']
del indf['timestamp_first_active']

del infile
for encoder in encoders:
	indf[encoder[1]] = encoder[0].fit_transform(indf[encoder[1]])
indf = np.array(indf)
indf = np.nan_to_num(indf)
predicted = forest.predict(indf)
counts = defaultdict(int)
for item in predicted:
	counts[item] += 1
countslist = list()
for element in counts:
	item = (element, counts[element])
	countslist.append(item)
countslist = sorted(countslist, key=itemgetter(1), reverse=True)
counts = list
print "id,country"
for count in countslist:
	for i in range(len(predicted)):
		if predicted[i] == count[0]:
			print ids[i] + ',' + predicted[i]
			
elapsed = clock() - current
print elapsed

