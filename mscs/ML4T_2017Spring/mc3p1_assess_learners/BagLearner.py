"""

Bagging Learner by Qi Wang, 2017.02.

One of the ensemble learning ideas:
  * Bagging: on resample data but weight components with their performance;
  * Boosting: re-weight incorrect samples for components;
  * Random Learning: data unchanged but with randomized method.

BagLearner is Bagging + Random Learning (RTLearner).

"""

import numpy as np
import RTLearner as rtl
import random as rd

class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.sub_learners = []

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term
        sample_cnt = dataX.shape[0]
        for i in range(0, self.bags):
            sample = np.random.choice(sample_cnt, sample_cnt)
            l = self.learner(**self.kwargs)
            l.addEvidence(dataX[sample,:], dataY[sample])
            self.sub_learners.append(l)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        predY = np.zeros([points.shape[0]])
        for l in self.sub_learners:
            predY += l.query(points) # get the predictions
        predY = predY * 1.0 / self.bags
        return predY

    def author(self):
      return 'qwang378'

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
