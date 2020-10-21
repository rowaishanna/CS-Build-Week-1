import numpy as np

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    def __str__(self):
        thr = f'THRESHOLD: {self.threshold}'
        pc = f'PREDICTED_CLASS: {self.predicted_class}'
        fi = f'FEATURE_INDEX: {self.feature_index}'
        l = f'LEFT: {self.left}'
        r = f'RIGHT: {self.right}'
        return f'{thr}, {pc}, {fi}, {l}, {r}'


class DecisionTreeClassifier:

    # max depth determines how many layers the tree will have.
    # when instantiating, it needs at least one layer to run
    def __init__(self, max_depth=0):
        self.max_depth = max_depth
        self.tree = None

    def __str__(self):
        return f'{self.tree}'
#_____________FIT_________________________
   
    def fit(self, X, y):
        '''
        Method which takes in two variables, X and y where
        X is a set of features and y is the target variable.
        Fitting the decision tree with this input data allows us to make
        predictions for a target (y) with new data (X).
        The decision tree will be built by finding the optimal features and thresholds 
        within this data.
        '''
        self.n_classes = len(set(y)) # creates a set of the unique possible outcomes
        self.n_features = X.shape[1] # the number of features in the given data
        self.tree = self._grow_tree(X, y) # creates a decision tree from X and y
        ### go to the _create_tree function...


#___________________________GROW_TREE___________________________

    def _grow_tree(self, X, y, depth=0):
        '''
        
        '''
        # Get a list of how many samples we have per class left to sort through
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]

        # get the index of the list that holds the maximum value in the list num_sample_per_class
        # returns an integer
        predicted_class = np.argmax(num_samples_per_class)

        # instantiate a node that has the index of the predicted class as its predicted class
        node = Node(predicted_class=predicted_class)
        

        # the following creates the layers/branches of the tree
        # in a recursive fashion
        if depth < self.max_depth:
            # find the index and threshold where we should split the tree
            # using _best_split
            idx, thr = self._best_split(X,y)
            ### go to the best_split function

            if idx is not None:
                # creates a new list of values from the current feature
                # that are less than the threshold
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left] # what does the ~ do?
                node.feature_index = idx
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

        # if depth == the max depth, we return the final node in the sequence
        return node


#___________________________BEST_SPLIT__________________________

    def _best_split(self, X, y):
        '''
        This finds the optimal feature and threshold at which to split the tree
        '''
        # get the size/length of y
        y_size = y.size
        # if our data is only one observation
        if y_size <= 1:
            # we can't make a prediction
            return None, None
            ### return to _grow_tree

        # Otherwise, get a list that will be used to count how many ocurrences we have of each class
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]

        # as we iterate through the different combinations below,
        # it will return the best gini
        best_gini = 1.0 - sum((n / y_size)**2 for n in num_parent)        
        best_idx, best_thr = None, None
        
        # now we loop through features
        # by their index
        for idx in range(self.n_features):
            # create an array of the potential thresholds
            thresholds = X[:, idx]

            # create an array of the potential classes
            classes = y

            # a list that will be used to count how many ocurrences we have of each class in the left node
            num_left = [0] * self.n_classes

            # a list that will be used to count how many ocurrences we have of each class in the right node
            num_right = num_parent.copy()


            for i in range(1, y_size):
                c = classes[i - 1] # this uses the list classes from above to return a single class, or answer (a class is the y or target value)
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes)) # returns the gini impurity for the left fork 
                gini_right = 1.0 - sum((num_right[x] / (y_size - i))**2 for x in range(self.n_classes)) # returns the gini impurity for the right fork
                gini = (i * gini_left + (y_size - i) * gini_right) / y_size # this finds the gini impurity

                if thresholds[i] == thresholds[i - 1]:
                    continue
                
                # update the index and the best gini value
                # if there is a better gini value than the previous gini value
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        # returns the index of the optimal threshold at which to split the node
        return best_idx, best_thr
        ### return to _grow_tree


#___________________________PREDICT___________________________

    def predict(self, X):
        # returns predictions using the helper function _predict
        return [self._predict(inputs) for inputs in X]


#___________________________PREDICT_HELPER___________________________

    def _predict(self, inputs):
        '''
        Traverses the binary tree until it finds the predicted class
        that is most closely associated with the given features 
        '''
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class