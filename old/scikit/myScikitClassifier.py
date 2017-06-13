from scipy.spatial import distance

"""
create my own nearest neighbor classifier
"""
class KNNeighbors():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = [self.closest(data) for data in x_test]
		return predictions
	
	def closest(self, data):
		nearest = distance.euclidean(data, self.x_train[0])
		nearestIndex = 0
		for i in range(1, len(self.x_train)):
			dist = distance.euclidean(data, self.x_train[i])
			if dist < nearest:
				nearest = dist
				nearestIndex = i
		#return output of best input
		return self.y_train[nearestIndex]
		

"""
use demo code on my own classifier
"""
from sklearn import datasets
iris = datasets.load_iris()

#using iris database which is pretty popular
x = iris.data
y = iris.target #f(x) = y

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

clfr = KNNeighbors()

#train classifier on given input/output
clfr.fit(x_train, y_train)
#generate predicted output of x-test data from our classifier
predictions = clfr.predict(x_test)

from sklearn.metrics import accuracy_score
#compare the output of the x-test data to the expected output of y-test
print(accuracy_score(y_test, predictions)) 
