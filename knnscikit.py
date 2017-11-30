# ============================== loading libraries ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import cross_val_score
# =============================================================================================
#					Part I
# =============================================================================================

# ============================== data preprocessing ===========================================
# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# loading training data
df = pd.read_csv('iris.data', header=None, names=names)
print(df.head())

# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4]) 	# end index is exclusive
y = np.array(df['class']) 	# showing you two ways of indexing a pandas df

# split into train and test

# ============================== KNN with k = 3 ===============================================
print('testing acc')
totalAcc = 0
for k in range(1,15):	
	for i in range(0,200):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
		# instantiate learning model (k = 3)
		knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')

		# fitting the model
		knn.fit(X_train, y_train)

		# predict the response
		pred = knn.predict(X_test)

		# evaluate accuracy
		acc = accuracy_score(y_test, pred, normalize=True)
		totalAcc += acc
		# acc = knn.score(X_test, y_test) 
		# print('\nThe accuracy of the knn classifier for k =' + repr(k) +  'is ' + repr(acc))
		# print(repr(acc))
		del knn
		del pred
		del acc
		del X_train
		del X_test
		del y_train
		del y_test
	totalAcc = totalAcc / 200
	print(repr(totalAcc))
	totalAcc = 0
print('training acc')
totalAcc = 0
for k in range(1,15):	
	for i in range(0,200):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
		# instantiate learning model (k = 3)
		knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')

		# fitting the model
		knn.fit(X_train, y_train)

		# predict the response
		pred = knn.predict(X_train)

		# evaluate accuracy
		acc = accuracy_score(y_train, pred, normalize=True)
		totalAcc += acc
		# acc = knn.score(X_test, y_test) 
		# print('\nThe accuracy of the knn classifier for k =' + repr(k) +  'is ' + repr(acc))
		# print(repr(acc))
		del knn
		del pred
		del acc
	totalAcc = totalAcc / 200
	print(repr(totalAcc))
	totalAcc = 0


# ============================== parameter tuning =============================================
# creating odd list of K for KNN
# myList = list(range(0,50))
# neighbors = list(filter(lambda x: x % 2 != 0, myList))

# # empty list that will hold cv scores
# cv_scores = []

# # perform 10-fold cross validation
# for k in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#     cv_scores.append(scores.mean())

# # changing to misclassification error
# MSE = [1 - x for x in cv_scores]

# # determining best k
# optimal_k = neighbors[MSE.index(min(MSE))]
# print('\nThe optimal number of neighbors is %d.' % optimal_k)

# # plot misclassification error vs k 
# plt.plot(neighbors, MSE)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Misclassification Error')
# plt.show()