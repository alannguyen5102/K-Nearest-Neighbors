import random
import math
import operator
import csv
import sys
from collections import Counter

#Unweighted K N N

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

	for k in range(len(neighbors)):
		targets.append(neighbors[k][-1])

	voted = Counter(targets).most_common(1)[0][0]
	return voted


def main():
	training = []


	testing = []
	predictions = []
	kFoldPercent = 0.80
	totalAccuracy = 0.0
	totalTests = 1
	k = 7
	

	#Tests a range of different neighbors
	for k in range(7, 8):
		print(repr(k) + ' nearest neighbors')


		#Does totalTests number of tests for an averge
		for x in range(0, totalTests):

			#trains and tests with a new fold in data everytime for good measures
			train(sys.argv[1], kFoldPercent, training, testing)

			#for each data in testing, classify it and measure accuracy
			for i in range(len(testing)):
				result = classify(training, testing[i], k)
				predictions.append(result)
				if (repr(result) != repr(testing[i][-1])):
					wrong = "wrong"
				else:
					wrong = "right"
				print(wrong + '> predicted=' + repr(result) + ', actual=' + repr(testing[i][-1]))
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

			for i in range(len(training)):
				#only use the training data
				result = classify(training, training[i], k)
				predictions.append(result)
				if (repr(result) != repr(training[i][-1])):
					wrong = "wrong"
				else:
					wrong = "right"
				print(wrong + '> predicted=' + repr(result) + ', actual=' + repr(training[i][-1]))
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










