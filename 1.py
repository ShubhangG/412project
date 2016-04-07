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
files = ('data/train_users_2.csv', 'data/age_gender_bkts.csv', 'data/sessions.csv' )
# s = pd.Series([7, 'Heisenberg', 3.14, -17891231, 'Happy eating!'], index=['A','B', 'C', 'D', 'E'])
# print s['A']
# d = {'Chicago': 1000, 'New York': 1300, 'Austin':200}
# cities = pd.Series(d)
# # print cities
# print cities[['Chicago', 'New York']]
# print cities[cities < 1100]
# print 'Chicago' in cities
# print 'Champaign' in cities
# data = {'year': [2010, 2011, 2012], 'team': ['Bears', 'Bears', 'Packers'], 'wins': [11, 8, 10], 'losses': [5,8,6]}
# football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'losses'])
# print football
# train_users_csv = pd.read_csv('data/train_users_2.csv')
files2 = list()
frames=list()
for f in files:
	curr = pd.read_csv(f)
	
	df = pd.DataFrame(curr)
	
	del curr
	# print df
	df2 = copy(df)
	del df2['id']
	del df2['date_account_created']
	del df2['date_first_booking']
	del df2['timestamp_first_active']
	labels = df2['country_destination']

	del df2['country_destination']
	# print df2
	# print labels
	# break
	# X_train_shuffled = pd.DataFrame()
	# X_test_shuffled = pd.DataFrame()
	# y_train_shuffled = pd.DataFrame()
	# y_test_shuffled = pd.DataFrame()
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
	# print df2.gender
	for encoder in encoders:
		df2[encoder[1]] = encoder[0].fit_transform(df2[encoder[1]])
	# df2['gender'] = encoders[0][0].fit_transform(df2.gender)

	# print df2
	# quit()
	# df2 = pd.get_dummies(df2, dummy_na=True)
	df2 = np.array(df2)
	# df2 = df2[np.logical_not(np.isnan(df2))]
	df2 = np.nan_to_num(df2)
	# print df2.shape
	# quit()
	# labels = labels[1:10000]
	sss = cross_validation.StratifiedShuffleSplit(labels, 1, test_size=.3, train_size=.6, random_state=0)
	for train_index, test_index in sss:
		# print len(df2)
		# print train_index
		# print len(train_index)
		# quit()
		X_train_shuffled, X_test_shuffled = df2[train_index], df2[test_index]
		y_train_shuffled, y_test_shuffled = labels[train_index], labels[test_index]
		# X_train_shuffled.append(df2[train_index])
		# X_test_shuffled.append(df2[test_index])
		# y_train_shuffled.append(labels[train_index])
		# y_test_shuffled.append(labels[test_index])
	# print X_train_shuffled, y_train_shuffled
	# quit()
	# k = KMeans()
	# k = k.fit(X_train_shuffled, y_train_shuffled)
	# print k.score(X_test_shuffled, y_test_shuffled)

	# quit()

	forest = RandomForestClassifier(n_estimators=10)
	forest = forest.fit(X_train_shuffled, y_train_shuffled)
	# print forest.score(X_test_shuffled, y_test_shuffled)
	# print forest.predict(X_test_shuffled)

	infile = pd.read_csv('data/test_users.csv')
	indf = pd.DataFrame(infile)
	ids = indf['id']
	del indf['id']
	del indf['date_account_created']
	del indf['date_first_booking']
	del indf['timestamp_first_active']
	# labels = indf['country_destination']

	# del indf['country_destination']


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
				# quit()

	# dummy = pd.get_dummies(df2[1:10000])
	# labels = labels[1:10000]
	# print X_train_shuffled.shape, y_train_shuffled.shape
	# quit()
	# dummy = pd.get_dummies(df2)
	# del df
	# print dummy
	# dummyl = list(dummy)
	# a = np.array(dummy)
	# a = np.array(df2)
	# X_train, X_test, y_train, y_test = cross_validation.train_test_split(a, labels, test_size=.4, random_state=0)
	# print X_train.shape, y_train.shape
	# print X_train
	# print y_train
	# break
	# print X_test.shape, y_test.shape
	# del dummy
	# print a
	# print a.shape
	elapsed = clock() - current
	# print elapsed

	break;



# df = pd.DataFrame(train_users_csv)
# print df