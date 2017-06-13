from sklearn import datasets
iris = datasets.load_iris()

#using iris database which is pretty popular
x = iris.data
y = iris.target #f(x) = y

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

from sklearn.neighbors import KNeighborsClassifier
clfr = KNeighborsClassifier()

#train classifier on given input/output
clfr.fit(x_train, y_train)
#generate predicted output of x-test data from our classifier
predictions = clfr.predict(x_test)

from sklearn.metrics import accuracy_score
#compare the output of the x-test data to the expected output of y-test
print(accuracy_score(y_test, predictions)) 
