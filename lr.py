from mnist import MNIST
import pandas as pd
import numpy as np
import pylab as plt

np.set_printoptions(threshold=np.nan)

class LinearRegressor():
    def __init__(self, mnist_directory, lr0=None, lr_dampener=None):
        self.mnist_directory = mnist_directory
        self.lr_dampener = lr_dampener
        self.holdout_data = None
        self.holdout_labels_original = None

        self.load_data(self.mnist_directory)
        print 'Loaded data...'
        if lr0 == None:
            self.lr0 = 0.02 / self.train_data.shape[0]
        else:
            self.lr0 = lr0

        self.weights = np.array([0] * self.train_data.shape[1])

    def load_data(self, mnist_directory):
        mndata = MNIST(mnist_directory)
        tr_data, tr_labels = mndata.load_training()
        te_data, te_labels = mndata.load_testing()
        self.train_data = np.array(tr_data)
        self.train_labels_original = np.array(tr_labels)
        self.test_data = np.array(te_data)
        self.test_labels_original = np.array(te_labels)

    def subset_data(self, train_amount, test_amount):
        if train_amount > 0:
            self.train_data = self.train_data[:train_amount]
            self.train_labels_original = self.train_labels_original[:train_amount]
        else:
            self.train_data = self.train_data[-train_amount:]
            self.train_labels_original = self.train_labels_original[-train_amount:]
        if test_amount > 0:
            self.test_data = self.test_data[:test_amount]
            self.test_labels_original = self.test_labels_original[:test_amount]
        else:
            self.test_data = self.test_data[-test_amount:]
            self.test_labels_original = self.test_labels_original[-test_amount:]
        print 'Subsetted data.'

    def reassign_labels_for_target(self, target):
        self.train_labels = [int(i == target) for i in self.train_labels_original]
        self.test_labels = [int(i == target) for i in self.test_labels_original]
        if self.holdout_labels_original is not None:
            self.holdout_labels = [int(i == target) for i in self.holdout_labels_original]
        print 'Reassigned labels for target value: {}'.format(target)


    def prefix_one(self, some_array):
        return [[1] + sr for sr in some_array]

    # This is all taken from CSE 250A
    def sigma(self, x, w):
        return 1 / (1 + np.exp(-1 * (np.dot(x, w))))

    def L(self, w, x, y):
        return np.sum(y * np.log(sigma(x, w)) + (1 - y) * np.log(sigma(-x, w)))

    def dl(self, w, x, y):
        to_return = np.array([0] * len(w))
        for t in xrange(x.shape[0]):
            xt = x[t, :]
            yt = y[t]
            temp = (yt - sigma(w, xt)) * xt
            to_return = np.add(to_return, temp)
        return to_return

    def assign_holdout(self, percent):
        percent /= 100.0
        num_held = int(self.train_data.shape[0] * percent)
        self.train_data = self.train_data[:-num_held]
        self.train_labels_original = self.train_labels_original[:-num_held]
        self.holdout_data = self.train_data[-num_held:]
        self.holdout_labels_original = self.train_labels_original[-num_held:]
        print 'Assigned holdout data'

    def update_learning_rate(self, iteration):
        if self.lr_dampener is not None:
            return self.lr0 / (1.0 + iteration / self.lr_dampener)
        else:
            return self.lr0

    def gradient_descent(self, iterations, anneal=True, log_rate=None):
        self.train_logs = []
        self.holdout_errors = []
        self.iter_steps = []
        # self.logs = []
        self.holdout_logs = []
        lr = self.lr0

        for t in xrange(iterations):
            if anneal:
                lr = self.update_learning_rate(t)
            prediction = self.sigma(self.train_data, self.weights)
            error = self.train_labels - prediction
            grad = np.dot(self.train_data.T, error)
            self.weights = np.add(self.weights, lr * grad)


            if log_rate is not None:
                if t % log_rate == 0:
                    self.iter_steps.append(t)
                    self.train_logs.append(self.evaluate(self.weights,
                                                         self.train_data,
                                                         self.train_labels)
                                                         )
                    if self.holdout_data is not None:
                        self.holdout_logs.append(self.evaluate(self.weights,
                                                               self.holdout_data,
                                                               self.holdout_labels)
                                                               )

    def evaluate(self, w, x, y):
        pred = np.round(self.sigma(x, w))
        return np.sum(np.abs(y - pred)) * 100 / x.shape[0]

    def train_on_number(self, num, iterations, log_rate=None, anneal=True):
        self.reassign_labels_for_target(num)
        self.gradient_descent(iterations, anneal=anneal, log_rate=log_rate)


RL = LinearRegressor('mnist', lr_dampener=100)
RL.subset_data(5000, -200)
RL.assign_holdout(10)

RL.reassign_labels_for_target(2)
RL.gradient_descent(1000, log_rate=100)

print RL.train_logs
print RL.holdout_logs
print RL.evaluate(RL.weights, RL.test_data, RL.test_labels)
