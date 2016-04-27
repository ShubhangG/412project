import numpy as np
import os
import pandas as pd
import math
import sys
from airbnb import *

#preprocessing without labels
def preprocessing4test(dataframe):
	dataframe = reduceLabels(dataframe)
	columns_to_encode = dataframe.columns.values.tolist()
	columns_to_encode = [x for x in columns_to_encode if not x.startswith('age')]
	encoder = Encoder(columns_to_encode)
	
	dataframe = encoder.encodeDataset(dataframe)
	numerical = np.array(dataframe)
	numerical = np.nan_to_num(numerical)
	return dataframe, numerical, encoder

#Step 1, seperate data into classes
def seperate_into_class(training_data):		#training data passsed in as a pandas dataframe
	seperated = {}
	cleaned_data= {}
	numeric_data = {}
	labels = {}
	encoder = {}
	#seperate data into each of the classes- which are the countries in this case
	seperated['US'] = training_data.loc[training_data['country_destination'] == 'US']
	seperated['AU'] = training_data.loc[training_data['country_destination'] == 'AU']
	seperated['CA'] = training_data.loc[training_data['country_destination'] == 'CA']
	seperated['DE'] = training_data.loc[training_data['country_destination'] == 'DE']
	seperated['ES'] = training_data.loc[training_data['country_destination'] == 'ES']
	seperated['FR'] = training_data.loc[training_data['country_destination'] == 'FR']
	seperated['GB'] = training_data.loc[training_data['country_destination'] == 'GB']
	seperated['IT'] = training_data.loc[training_data['country_destination'] == 'IT']
	seperated['NL'] = training_data.loc[training_data['country_destination'] == 'NL']
	seperated['PT'] = training_data.loc[training_data['country_destination'] == 'PT']

	#now preprocess the data seperately
	for country, data in seperated.iteritems():
		cleaned_data[country], numeric_data[country], labels[country], encoder[country] =  preprocessing(data)

	return numeric_data

def guassian_pdf(x, mean, std):										#pdf formula	
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
	return (1 / (math.sqrt(2*math.pi) * std)) * exponent

def calculate_summary(numeric_data):								#summarizes all the important info about our training data in terms of mean and standard dev  
	mean_of_data = {}
	std_of_data = {}
	summaries = {}

	for country, data in numeric_data.iteritems():
		mean_of_data[country] = numeric_data[country].mean(axis = 0)

	for country, data in mean_of_data.iteritems():
		std_of_data[country] = numeric_data[country].std(axis = 0)

	return mean_of_data, std_of_data


def class_prob(input_vector, mean, std):									#Assumption- attributes- Age, Gender, etc. are independent of each other 
	probability = {}														#so the probabilities multiply, so the probability of a class is													
	for attribute in range(0,len(input_vector)):							#product of probabilities of its attributes
		probability[attribute] = guassian_pdf(input_vector[attribute], mean[attribute], std[attribute])

	class_probability = 1.0
	for attribute, pdf in probability.iteritems():
		class_probability = class_probability*pdf

	return class_probability


def Predict(input_vector, mean_data, std_data): 							#gives us the predicted country 
 	class_pdf = {}
	for country, means in mean_data.iteritems():
		class_pdf[country] = class_prob(input_vector, means, std_data[country])

	return max(class_pdf, key=lambda i: class_pdf[i])
	
def Setup_Naive_Bayes(training_data): 
	#Seperate data by classes
	numeric_data = seperate_into_class(training_data)
	
	#calculate summary with standard deviation and mean
	mean_data, std_data = calculate_summary(numeric_data)

	return mean_data, std_data


def Naive_Bayes_Predictor(test_data, mean_data, std_data):					#Wraps up everything
	cleaned_test, numerical, encoder = preprocessing4test(test_data)
	labels = []
	for row_id in range(0, len(numerical)):
		labels.append(Predict(numerical[row_id],mean_data,std_data))
	return labels



