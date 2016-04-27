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
		self.clean = self.clean[:,0:11]
		self.centers = None
	
	def initCentroids(self):
		categories = set(np.unique(self.labels))
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
				if x[0][11] == curr:
				# if self.data[i][11] == curr:
					centroids.append(x[0][0:11])
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
			centers = self.centers
			data = curdata
		else:
			centers = candidate_centers
			data = self.clean
		clusters = {}
		results = {}
		# for x in self.clean:
		for x in data:
			bestcenter = min([(i[0], np.linalg.norm(x-centers[i[0]])) \
				for i in enumerate(centers)], key=lambda t:t[1])[0]
			try:
				clusters[bestcenter].append(x)
			except KeyError:
				clusters[bestcenter] = [x]

		return clusters

	def reevaluate(self, old_centers, clusters):
		new_centers = []
		keys = sorted(clusters.keys())
		for k in keys:
			new_centers.append(np.mean(clusters[k], axis=0))
		return new_centers

	def hasConverged(self, centers, old_centers):
		return (set([tuple(a) for a in centers]) == set([tuple(a) for a in old_centers]))

	def predict(self, test_data, test_labels):
		cleaned = test_data[:,0:11]
		ids = [i for i in range(len(cleaned))]
		# print len(cleaned)
		# print len(ids)
		clusters = self.clusterPoints(curdata=cleaned, test=True)
		for i in range(len(clusters)):
			print "Elements in {}".format(i)
			print len(clusters[i])

		print len(clusters)


	def findCenters(self):
		old_centers = random.sample(self.data, self.k)
		# new_centers = random.sample(self.data, self.k)
		# old_centers = self.initCentroids()
		new_centers = self.initCentroids()
		# print old_centers
		# print new_centers
		i = 0
		while i < 1 and not self.hasConverged(new_centers, old_centers):
			i += 1
			print "Iteration " + str(i)
			old_centers = new_centers
			clusters = self.clusterPoints(candidate_centers=new_centers)
			new_centers = self.reevaluate(old_centers, clusters)

		self.centers = new_centers
		return(new_centers, clusters)