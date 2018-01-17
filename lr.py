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
        self.holdout_data_original = None
        self.holdout_labels_original = None
        self.target = None
        self.load_data(self.mnist_directory)
        print 'Loaded data...'
        if lr0 == None:
            self.lr0 = 0.001 / self.train_data_original.shape[0]
        else:
            self.lr0 = lr0

        self.initialize_weights()

    def initialize_weights(self):
        self.weights = np.array([0] * self.train_data_original.shape[1])

    def dl1(self, w):
        return np.sign(w)

    def dl2(self, w):
        return 2 * w

    def load_data(self, mnist_directory):
        mndata = MNIST(mnist_directory)
        tr_data, tr_labels = mndata.load_training()
        te_data, te_labels = mndata.load_testing()
        train_temp = np.array(tr_data)
        self.train_data_original = np.concatenate(
                                        (np.ones((train_temp.shape[0], 1)),
                                         train_temp
                                        ), axis=1
                                        )
        self.train_labels_original = np.array(tr_labels)
        test_temp = np.array(te_data)
        self.test_data_original = np.concatenate(
                                        (np.ones((test_temp.shape[0], 1)),
                                         test_temp
                                        ), axis=1
                                        )
        self.test_labels_original = np.array(te_labels)

    def subset_data(self, train_amount, test_amount):
        if train_amount > 0:
            self.train_data_original = self.train_data_original[:train_amount]
            self.train_labels_original = self.train_labels_original[:train_amount]
        else:
            self.train_data_original = self.train_data_original[-train_amount:]
            self.train_labels_original = self.train_labels_original[-train_amount:]
        if test_amount > 0:
            self.test_data_original = self.test_data_original[:test_amount]
            self.test_labels_original = self.test_labels_original[:test_amount]
        else:
            self.test_data_original = self.test_data_original[-test_amount:]
            self.test_labels_original = self.test_labels_original[-test_amount:]
        print 'Subsetted data.'

    def reassign_labels_for_target(self, target, not_target):
        self.target = target
        self.not_target = not_target
        new_set = [target, not_target]
        indices = [i in new_set for i in self.train_labels_original]
        self.train_data = np.array([self.train_data_original[i] for i in xrange(len(self.train_data_original)) if indices[i]])
        self.train_labels = np.array([int(self.train_labels_original[i] == target) for i in xrange(len(self.train_labels_original)) if indices[i]])
        indices = [i in new_set for i in self.test_labels_original]
        self.test_data = np.array([self.test_data_original[i] for i in xrange(len(self.test_data_original)) if indices[i]])
        self.test_labels = np.array([int(self.test_labels_original[i] == target) for i in xrange(len(self.test_labels_original)) if indices[i]])

        if self.holdout_labels_original is not None:
            indices = [i in new_set for i in self.holdout_labels_original]
            self.holdout_data = np.array([self.holdout_data_original[i] for i in xrange(len(self.holdout_data_original)) if indices[i]])
            self.holdout_labels = np.array([int(self.holdout_labels_original[i] == target) for i in xrange(len(self.holdout_labels_original)) if indices[i]])
        print 'Reassigned labels for target value: {}'.format(target)

    def sigma(self, x, w):
        return 1 / (1 + np.exp(-1 * (np.dot(x, w))))

    def L(self, w, x, y):
        return np.sum(y * np.log(sigma(x, w)) + (1 - y) * np.log(sigma(-x, w)))

    def dL(self, w, x, y):
        prediction = self.sigma(x, w)
        error = y - prediction
        return np.dot(np.transpose(x), error)

    def norm_loss_function(self, w, x, y):
        return (-1.0 / x.shape[0]) * \
               np.sum(y * np.log(self.sigma(x, w)) + (1 - y) * \
               np.log(self.sigma(-x, w)))

    def assign_holdout(self, percent):
        percent /= 100.0
        num_held = int(self.train_data_original.shape[0] * percent)
        self.train_data_original = self.train_data_original[:-num_held]
        self.train_labels_original = self.train_labels_original[:-num_held]
        self.holdout_data_original = self.train_data_original[-num_held:]
        self.holdout_labels_original = self.train_labels_original[-num_held:]
        print 'Assigned holdout data'

    def update_learning_rate(self, iteration):
        if self.lr_dampener is not None:
            return self.lr0 / (1.0 + iteration / self.lr_dampener)
        else:
            return self.lr0

    def gradient_descent(self, iterations, anneal=True, log_rate=None,
                         l1=False, l2=False, lamb=None):
        if l1 and l2:
            raise ValueError('Only do l1 or l2')
        if (l1 or l2) and lamb is None:
            raise ValueError('Specify lambda if l1 and l2 flags on.')
        self.iter_steps = []

        self.train_logs = []
        self.test_logs = []
        self.holdout_logs = []

        self.train_loss = []
        self.holdout_loss = []
        self.test_loss = []
        lr = self.lr0

        for t in xrange(iterations):
            if anneal:
                lr = self.update_learning_rate(t)
            grad = self.dL(self.weights, self.train_data, self.train_labels)
            # print grad
            if l1:
                grad -= lamb * self.dl1(self.weights)
            if l2:
                grad -= lamb * self.dl2(self.weights)
            self.weights = np.add(self.weights, lr * grad)
            # print grad
            # print '----'
            if log_rate is not None:
                if t % log_rate == 0:
                    self.iter_steps.append(t)
                    self.train_logs.append(self.evaluate(self.weights,
                                                         self.train_data,
                                                         self.train_labels)
                                                         )
                    self.train_loss.append(self.norm_loss_function(self.weights,
                                                                   self.train_data,
                                                                   self.train_labels)
                                                                   )
                    self.test_logs.append(self.evaluate(self.weights,
                                                        self.test_data,
                                                        self.test_labels)
                                                        )
                    self.test_loss.append(self.norm_loss_function(self.weights,
                                                                   self.test_data,
                                                                   self.test_labels)
                                                                   )
                    if self.holdout_data is not None:
                        self.holdout_logs.append(self.evaluate(self.weights,
                                                               self.holdout_data,
                                                               self.holdout_labels)
                                                               )
                        self.holdout_loss.append(self.norm_loss_function(self.weights,
                                                                       self.holdout_data,
                                                                       self.holdout_labels)
                                                                       )

    def evaluate(self, w, x, y):
        pred = np.round(self.sigma(x, w))
        return np.sum(np.abs(y - pred)) * 100.0 / x.shape[0]

    def train_on_number(self, num, iterations, log_rate=None, anneal=True):
        self.reassign_labels_for_target(num)
        self.gradient_descent(iterations, anneal=anneal, log_rate=log_rate)

    def plot_logs(self):
        plt.plot(self.iter_steps, self.train_logs, label='Training Data')
        plt.plot(self.iter_steps, self.holdout_logs, label='Holdout Data')
        plt.plot(self.iter_steps, self.test_logs, label='Test Data')
        plt.ylabel('Percent misclassified')
        plt.xlabel('Iterations')
        plt.title('Gradient descent for character: {} vs {}'.format(self.target,
                                                                    self.not_target))
        plt.legend(loc='upper right')
        plt.show()

        plt.plot(self.iter_steps, self.train_loss, label='Training Data')
        plt.plot(self.iter_steps, self.holdout_loss, label='Holdout Data')
        plt.plot(self.iter_steps, self.test_loss, label='Test Data')
        plt.ylabel('Loss Function')
        plt.xlabel('Iterations')
        plt.title('Gradient descent for character: {} vs {}'.format(self.target,
                                                                    self.not_target))
        plt.legend(loc='upper right')
        plt.show()
