from sklearn.datasets import load_iris
from dec_tree import DecisionTreeClassifier
from sklearn import tree

# load the iris dataset
dataset = load_iris()

# set X and y variables
X, y = dataset.data, dataset.target
print('---------------------------------------------')
print(f'APPROPRIATE X, y DATATYPES: {type(X)}')
print('---------------------------------------------')
# create a new isntance of the DecisionTreeClassifier object
clf = DecisionTreeClassifier(max_depth=5)

# call the fit method on that object 
clf.fit(X, y)
print('')
print('----------------PREDICTIONS-----------------------')
print('')
print('---------------------------------------------')
inputs = [[1, 1.5, 5, 1.5]]
print(f'INPUTS: {inputs}')
print(f'OUR MODEL PREDICTION: {clf.predict(inputs)}')

clf2 = tree.DecisionTreeClassifier(max_depth=5)
clf2.fit(X, y)

print(f'SCIKITLEARN MODEL PREDICTION: {clf2.predict(inputs)}')
print('---------------------------------------------')