from mnist import MNIST
import pandas as pd
import numpy as np
import pylab as plt
from helper import *

# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=100)

class SoftMax():
    def __init__(self, mnist_directory, lr0=None, lr_dampener=None):
        self.mnist_directory = mnist_directory
        self.lr_dampener = lr_dampener
        self.holdout_data = None
        self.holdout_labels_original = None
        self.target = None
        self.load_data(self.mnist_directory)
        print 'Loaded data...'
        if lr0 == None:
            self.lr0 = 0.001 / self.train_data.shape[0]
        else:
            self.lr0 = lr0

        self.initialize_weights()

    def initialize_weights(self):
        self.weights = np.zeros((self.train_data.shape[1],
                                 self.num_categories
                                 ))

    def dl1(self, w):
        return np.sign(w)

    def dl2(self, w):
        return 2 * w

    def get_regularized_labels(self, labels):
        potential_vals = list(set(labels))
        potential_vals.sort()
        return np.array([[int(l == p) for p in potential_vals] for l in labels])

    def load_data(self, mnist_directory):
        mndata = MNIST(mnist_directory)
        tr_data, tr_labels = mndata.load_training()
        te_data, te_labels = mndata.load_testing()
        train_temp = np.array(tr_data)
        self.train_data = np.concatenate(
                                        (np.ones((train_temp.shape[0], 1)),
                                         train_temp
                                        ), axis=1
                                        )
        self.train_labels = np.array(tr_labels)
        test_temp = np.array(te_data)
        self.test_data = np.concatenate(
                                        (np.ones((test_temp.shape[0], 1)),
                                         test_temp
                                        ), axis=1
                                        )
        self.test_labels = np.array(te_labels)
        self.num_categories = len(list(set(self.train_labels)))
        self.possible_categories = list(set(self.train_labels))
        self.possible_categories.sort()

    def subset_data(self, train_amount, test_amount):
        if train_amount > 0:
            self.train_data = self.train_data[:train_amount]
            self.train_labels = self.train_labels[:train_amount]
        else:
            self.train_data = self.train_data[-train_amount:]
            self.train_labels = self.train_labels[-train_amount:]
        if test_amount > 0:
            self.test_data = self.test_data[:test_amount]
            self.test_labels = self.test_labels[:test_amount]
        else:
            self.test_data = self.test_data[-test_amount:]
            self.test_labels = self.test_labels[-test_amount:]
        print 'Subsetted data.'

    def prefix_one(self, some_array):
        return [[1] + sr for sr in some_array]

    def softmax(self, x, w):
        dot_exp = np.exp(np.dot(x, w))
        summed = np.sum(dot_exp, axis=1)
        summed = np.reshape(summed, (dot_exp.shape[0], 1))
        summed = np.repeat(summed, dot_exp.shape[1], axis=1)
        return (dot_exp / (1.0 * summed))

    def L(self, w, x, y):
        rvals = self.get_regularized_labels(y)
        scores = self.softmax(x, w)
        return -1 * np.sum(rvals * np.log(scores))
        # return np.sum(y * np.log(self.softmax(x, w)) + (1 - y) * np.log(self.softmax(-x, w)))

    def norm_loss_function(self, w, x, y):
        y = self.get_regularized_labels(y)
        return (-1.0 / (x.shape[0] * w.shape[1])) * np.sum(y * self.softmax(x, w))

    def dl(self, w, x, y):
        difference = (self.get_regularized_labels(y) - self.softmax(x, w))
        return np.dot(np.transpose(x), difference)

    def assign_holdout(self, percent):
        percent /= 100.0
        num_held = int(self.train_data.shape[0] * percent)
        self.train_data = self.train_data[:-num_held]
        self.train_labels = self.train_labels[:-num_held]
        self.holdout_data = self.train_data[-num_held:]
        self.holdout_labels = self.train_labels[-num_held:]
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
        self.weight_lengths = []
        lr = self.lr0

        for t in xrange(iterations):
            if anneal:
                lr = self.update_learning_rate(t)
            grad = self.dl(self.weights, self.train_data, self.train_labels)

            if l1:
                grad -= lamb * self.dl1(self.weights)
            if l2:
                grad -= lamb * self.dl2(self.weights)

            self.weights = np.add(self.weights, lr * grad)
            if log_rate is not None:
                if t % log_rate == 0:
                    self.weight_lengths.append(np.sum(is_close(self.weights, 0, 0.001)))
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
        ind = np.argmax(self.softmax(x, w), axis=1)
        pred = [self.possible_categories[i] for i in ind]
        return 100.0 - 100.0 * np.sum((pred != y).astype(int)) / (1.0 * x.shape[0])

    def train_on_number(self, num, iterations, log_rate=None, anneal=True):
        self.reassign_labels_for_target(num)
        self.gradient_descent(iterations, anneal=anneal, log_rate=log_rate)

    def plot_logs(self):
        plt.plot(self.iter_steps, self.train_logs, label='Training Data')
        plt.plot(self.iter_steps, self.holdout_logs, label='Holdout Data')
        plt.plot(self.iter_steps, self.test_logs, label='Test Data')
        plt.ylabel('Percent classified correctly')
        plt.xlabel('Iterations')
        plt.title('Softmax Regression')
        plt.legend(loc='lower right')
        plt.show()

        plt.plot(self.iter_steps, self.train_loss, label='Training Data')
        plt.plot(self.iter_steps, self.holdout_loss, label='Holdout Data')
        plt.plot(self.iter_steps, self.test_loss, label='Test Data')
        plt.ylabel('Loss Function')
        plt.xlabel('Iterations')
        plt.title('Softmax Regression')
        plt.legend(loc='upper right')
        plt.show()
