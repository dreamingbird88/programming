"""

Random Tree Learner by Qi Wang, 2017.02.

    class Node(object):
        def __init__(self, val, split=None, left=None, right=None):
            self.val = val
            self.split = split
            self.left = left
            self.right = right

"""

import numpy as np
import random as rd

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def build_tree(self, dataX, dataY):
        sample_size = dataX.shape[0] 
        if sample_size <= self.leaf_size or all(y==dataY[0] for y in dataY):
            return [dataY.mean()]
        retry = True
        while retry: 
            # randomly select feature
            i = rd.randint(0, self.feature_size-1)
            # randomly select value
            j1 = rd.randint(0, sample_size-1)
            j2 = rd.randint(0, sample_size-1)
            v = (dataX[j1,i] + dataX[j2,i]) / 2.0
            less = dataX[:,i] <= v
            size = sum(less)
            retry = size == 0 or size == sample_size
        left = self.build_tree(dataX[less], dataY[less])
        right = self.build_tree(dataX[~less], dataY[~less])
        #if self.verbose: print [i, v, size, sample_size - size]
        # split feature, split_value, left, right
        return [i, v, left, right]

    def query_tree(self, tree, data):
        # leaf node
        if len(tree) == 1:
            return tree * data.shape[0]
        predY = np.zeros([data.shape[0]])
        left = data[:, tree[0]] <= tree[1]
        if sum(left) > 0:
            predY[left] = self.query_tree(tree[2], data[left])
        if sum(~left) > 0:
            predY[~left] = self.query_tree(tree[3], data[~left])
        return predY

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.feature_size = dataX.shape[1]
        self.random_tree = self.build_tree(dataX,dataY)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.query_tree(self.random_tree, points)

    def author(self):
      return 'qwang378'

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
