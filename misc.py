from sklearn import preprocessing
from copy import copy
import numpy as np
import random


class Encoder():
    def __init__(self, features):
        self.encoders = {}
        for feature in features:
            self.encoders[feature] = preprocessing.LabelEncoder()

    def encodeDataset(self, dataframe):
        for encoder in self.encoders:
            if encoder in dataframe:
                dataframe[encoder] = self.encoders[encoder].fit_transform(dataframe[encoder])
        return dataframe

    def decode(self, data, encoder):
        return self.encoders[encoder].inverse_transform(data)


class KMeans():
	def __init__(self, data=None, labels=None, k=0):
		self.data = data
		self.labels = labels
		self.k = k
		self.clean = copy(data)
		self.clean = self.clean[:,0:12]
		self.centers = None
	
	def initCentroids(self):
		categories = set(np.unique(self.labels[:,1]))
		centroids = []
		while categories:
			curr = categories.pop()
			while True:
				x = random.sample(self.data, 1)
				# print x[0]
				# print x[0][0:11]
				
				# print x
				# print x[0]
				# quit()
			# print "Looking for " + str(curr)
			# for i in range(len(self.data)):
				# print self.labels[i]
				# print i
				
				
				# print "Current = " + str(self.data[i])
				# print self.labels[i]
			
				if x[0][12] == curr:

				# if self.data[i][11] == curr:
					# print x[0]
					# print x[0][0:11]
					# print x[0][1:12]
					# quit()
					# centroids.append(x[0][0:11])
					centroids.append(x[0][1:12])
					# centroids.append(self.data[i])
					break
			# print centroids
			# print len(centroids)
		# quit()
		# self.clean = copy(self.data)
		# print self.clean[:10]
		# print self.data[:10]
		# del self.data['country_destination']
		# self.clean = self.data[:,0:11]
		return centroids
		# print self.old[:10,0:11]
		# print
		# print self.data[:10]
		# print self.old.shape

			
			# print categories
			# categories.remove(category)
		
		# print categories
		# print self.data


	def clusterPoints(self, curdata=None, candidate_centers=None, test=False, ids=None):
		if test == True:
			# centers = self.centers[:][1:2]
			arr = np.array(self.centers)
			if arr.shape == (12,12):

				centers = [a[1:12] for a in self.centers]
			else:
				centers = self.centers
			data = curdata
			# print "test len"
			# print len(centers)
			# quit()
			# print centers
			# print list(centers)
			# quit()
			# print self.centers[0]
			# print centers[0]
			# quit()
		else:
			# print candidate_centers[0]
			# quit()
			a = np.array(candidate_centers)
			# print a.shape
			# if a.shape == (12, 12):
				# print "Fuck"
			centers = candidate_centers
			# print "reg cand"
			# print len(candidate_centers)
			# centers = [a[0:11] for a in candidate_centers]
			# print "reg len"
			# print len(centers)
			# print centers[0]
			# quit()
			
			data = self.clean
			# print candidate_centers

		clusters = {}
		results = {}
		# for x in self.clean:
		# printed = False
		for x in data:
			y = x[1:12]
			# if not printed:
				# printed = True
				# print y
				# a = np.array(centers)
				# print a.shape
				# print centers[0]
			

			# bestcenter = min([(i[0], np.linalg.norm(x-centers[i[0]])) \
			# 	for i in enumerate(centers)], key=lambda t:t[1])[0]
			bestcenter = min([(i[0], np.linalg.norm(y-centers[i[0]])) \
				for i in enumerate(centers)], key=lambda t:t[1])[0]
			try:
				clusters[bestcenter].append(x)
			except KeyError:
				clusters[bestcenter] = [x]

		# print clusters[0][0]
		# quit()		
		return clusters

	def reevaluate(self, old_centers, clusters):
		new_centers = []
		keys = sorted(clusters.keys())
		for k in keys:
			
			# print clusters[k][0]
			# print clusters[k][0][0]
			# print clusters[k][1:5][1:12]
			# quit()
			new_centers.append(np.mean(clusters[k], axis=0))
		return new_centers

	def hasConverged(self, centers, old_centers):
		return (set([tuple(a) for a in centers]) == set([tuple(a) for a in old_centers]))

	def separateLabels(self, test_labels):
		output = []
		for i in range(12):
			output.append(set())
		
		i = 0
		for element in test_labels:
			i += 1
			output[element[1]].add(element[0])
		return output

	def predict(self, test_data, test_labels, validate=False):
		# cleaned = test_data[:,0:11]
		
		if test_data.shape[1] == 13:
			cleaned = test_data[:,0:12]
		else:
			cleaned = test_data
		ids = [i for i in range(len(cleaned))]
		# print len(cleaned)
		# print len(ids)
		clusters = self.clusterPoints(curdata=cleaned, test=True)
		if validate:
			return clusters
		total = 0
		correct = 0
		# print "Predicting"
		if not validate:
			labelsets = self.separateLabels(test_labels)
		confusion = np.zeros([12,12])
		output = []
		for i in range(len(clusters)):
		
			# print clusters
			# quit()
			# print "Testing category {}".format(i)
			category = 0
			category_correct = 0
			# print "Elements in {}".format(i)
			# print clusters[i][0]
			cur = 0
			if not validate:
				cat_len = len(clusters[i])
			stat = 0
			for current in clusters[i]:
				stat += 1
				cur += 1
				if not validate:
					progress = float(cur) / float(cat_len)
					# if stat % 1000 == 0:

						# print "Progress: {:.2f}%".format(progress*100)
				total += 1
				category += 1
				curid = current[0]
				output.append((curid, i))
				if validate:
					continue
				if curid in labelsets[i]:
					# print "Found one!"
					correct += 1
					category_correct += 1
					confusion[i,i] += 1
				else:
					for j in range(len(clusters)):
						if curid in labelsets[j]:
							confusion[i,j] += 1



			# curid = clusters[i][0][0]
				# for j in range(len(test_labels)):
					# print test_labels[j][0]
					# if test_labels[j][0] == curid:
						# if i == test_labels[j][1]:
							# correct += 1
							# category_correct += 1
							# print "Correct"
							# break
					# print "labeled = "
					# print test_labels[j][1]
					# quit()
			# print len(clusters[i])
			if not validate:

				cat_acc = float(category_correct) / float(category)
				# print len(clusters[i])
				# print "Category {} accuracy = {}".format(i, cat_acc)

			# quit()
		# print len(clusters)
		if validate:
			return output
		tot_acc = float(correct) / float(total)

		# print "Total accuracy = {}".format(tot_acc)
		# print confusion
		

	def initClusters(self):
		clusters = {}
		for x in self.data:
			try:
				clusters[x[12]].append(x[:12])
			except:
				clusters[x[12]] = [x[:12]]
				# clusters[x[12]]
		return clusters

	def findCenters(self):
		# old_centers = random.sample(self.data, self.k)
		# new_centers = random.sample(self.data, self.k)
		old_centers = self.initCentroids()
		new_centers = self.initCentroids()
		# print old_centers
		# print new_centers
		i = 0
		clusters = self.initClusters()
		new_centers = self.reevaluate(old_centers, clusters)
		# print clusters[0][:10]
		# quit()
		# print clusters.shape
		# quit()
		while i < 50 and not self.hasConverged(new_centers, old_centers):
			i += 1
			# print "Iteration " + str(i)
			a = np.array(old_centers)
			b = np.array(new_centers)
			# print a.shape, b.shape
			if a.shape != (12, 11):
				# print "wrong"
				old_centers = [c[1:12] for c in old_centers]
			if b.shape != (12, 11):
				# print "really wrong"
				new_centers = [c[1:12] for c in new_centers]
			old_centers = new_centers
			clusters = self.clusterPoints(candidate_centers=new_centers)
			new_centers = self.reevaluate(old_centers, clusters)
			# print clusters[0][:10]
			# quit()

		self.centers = new_centers
		return(new_centers, clusters)
