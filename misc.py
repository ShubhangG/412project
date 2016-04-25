from sklearn import preprocessing
from copy import copy
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
