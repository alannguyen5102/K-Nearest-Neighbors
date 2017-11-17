import random
import math
import operator
import csv
import sys
import pandas as pd 
from collections import Counter

def train(filename, split, training=[] , testing=[]):


	#opens csv file and converts to list
	with open(filename, 'rb') as file:
	    lines = csv.reader(file)
	    dataList = list(lines)
	    numAttributes = len(dataList[0]) - 1

	    #goes through all data in list
	    for x in range(len(dataList)-1):
	    	#converts the attributes to a floating point
	        for y in range(numAttributes):
	            dataList[x][y] = float(dataList[x][y])

	        #seperates a part of the data to be training and to be testing
	        if random.random() < split:
	            training.append(dataList[x])
	        else:
	            testing.append(dataList[x])
def calculateAccuracy(prediction, testedSet):
	
	#total is the number of correct predictions
	total = 0

	accuracy = 0.0

	#Goes through the predictions and compares it with the label of the set
	for i in range(len(testedSet)):
		if (prediction[i] == testedSet[i][-1]):
			total = total + 1

	#accuracy is total correct / total 
	accuracy = (float(total) / float(len(testedSet)))
	accuracy = accuracy * 100
	return accuracy

def getUnweightedDistance(point1, point2, l):
	
	#This allows the calucation of distance in numAttributes'th space
	distance = 0.0

	for i in range(l):
		distance += pow((point1[i] - point2[i]), 2)

	distance = math.sqrt(distance)

	return distance

def distance(neighborset,testset):
    dist = 1
    for x in range(len(neighborset)-1):
        dist += abs(neighborset[x] - testset[x])
    #print (dist)    
    return 1/dist

def classify(testedSet, point, k):
	#classifies a point

	#stores a list of distances
	euclideanDistance = []
	neighbors = []
	targets = []
	attributes = len(point) - 1
	for i in range(len(testedSet)):
		#adds each point of the tested set with the point
		d = getUnweightedDistance(point, testedSet[i], attributes)
		euclideanDistance.append((testedSet[i], d))

	#sorts the distances so shortest is front
	euclideanDistance.sort(key=operator.itemgetter(1))

	#puts only k shortest distances in neighbors
	for j in range(k):
		neighbors.append(euclideanDistance[j][0])

	classVotes = {}
	for i in range(len(neighbors)):
		response = neighbors[i][-1]
        #You need some way to define the weight here to decide the weight of each response
		weight = distance(neighbors[i],point)
		if response in classVotes:
			classVotes[response] += weight #changed 1 to weight
		else:
			classVotes[response] = weight #changed 1 to weight
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def main():
	training = []
	testing = []
	predictions = []
	kFoldPercent = 0.80
	totalAccuracy = 0.0
	totalTests = 50
	

#Tests a range of different neighbors
	print('all nearest neighbors')


	#Does totalTests number of tests for an averge
	for x in range(0, totalTests):

		#trains and tests with a new fold in data everytime for good measures
		train(sys.argv[1], kFoldPercent, training, testing)
		k = len(testing)	
		#for each data in testing, classify it and measure accuracy
		for i in range(len(testing)):
			result = classify(training, testing[i], k)
			predictions.append(result)
		accuracy = calculateAccuracy(predictions, testing)
		totalAccuracy += accuracy


		#resets the data for another fold
		del result
		del predictions[:]
		del training[:]
		del testing[:]

		accuracy = 0
	totalAccuracy = totalAccuracy / totalTests
	print('Testing Accuracy: ' + repr(totalAccuracy) + '%')

	totalAccuracy = 0

	#Does totalTests number of tests for an averge
	for x in range(0, totalTests):

		train(sys.argv[1], kFoldPercent, training, testing)
		k = len(training)
		for i in range(len(training)):
			#only use the training data
			result = classify(training, training[i], k)
			predictions.append(result)
		accuracy = calculateAccuracy(predictions, training)
		totalAccuracy += accuracy

		del result
		del predictions[:]
		del training[:]
		del testing[:]

		accuracy = 0
	totalAccuracy = totalAccuracy / totalTests
	print('Training Accuracy: ' + repr(totalAccuracy) + '%\n')

	totalAccuracy = 0
main()
