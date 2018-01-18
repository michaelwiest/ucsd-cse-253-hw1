from lr import *
from sm import *
from helper import *


# Part 1

# LR = LinearRegressor('mnist', lr_dampener=75)
# LR.subset_data(20000, -2000)
#
# LR.assign_holdout(10)
# LR.reassign_labels_for_target(2, 3)
# LR.gradient_descent(2000, log_rate=50) #, l1=True, lamb=1000)
# LR.plot_logs()
# w = LR.weights[1:]
#
#
# LR.initialize_weights()
# LR.reassign_labels_for_target(2, 8)
# LR.gradient_descent(2000, log_rate=50) #, l1=True, lamb=1000)
# LR.plot_logs()
# w2 = LR.weights[1:]
# mnist_printer(w)
# mnist_printer(w2)
# mnist_printer(w-w2)



# Part 2
LR = LinearRegressor('mnist', lr_dampener=75)
LR.subset_data(20000, -2000)
# lambdas = []
# l1_train_errors = []
# l2_train_errors = []
# l1_errors = []
# l2_errors = []
#
# l1_train_weights = []
# l2_train_weights = []
# for p in xrange(-2, 6):
#     lamb = 10 ** p
#     lambdas.append(lamb)
#
#     LR.initialize_weights()
#     LR.reassign_labels_for_target(2, 3)
#     LR.gradient_descent(1000, log_rate=50, l1=True, lamb=lamb)
#     # evaluation = LR.evaluate(LR.weights, LR.test_data, LR.test_labels)
#     l1_train_errors.append(LR.train_logs)
#     # l1_sum = sum([int(not is_close(r, 0, 0.001)) for r in LR.weights])
#     l1_errors.append(LR.evaluate(LR.weights, LR.test_data, LR.test_labels))
#     l1_train_weights.append(LR.weight_lengths)
#
#     LR.initialize_weights()
#     LR.gradient_descent(1000, log_rate=50, l2=True, lamb=lamb)
#     # evaluation = LR.evaluate(LR.weights, LR.test_data, LR.test_labels)
#     l2_train_errors.append(LR.train_logs)
#     l2_errors.append(LR.evaluate(LR.weights, LR.test_data, LR.test_labels))
#     # l2_sum = sum([int(not is_close(r, 0, 0.001)) for r in LR.weights])
#     l2_train_weights.append(LR.weight_lengths)
#
# for i in xrange(len(lambdas)):
#     plt.plot(LR.iter_steps, l1_train_errors[i], label='L1: Lambda = {}'.format(lambdas[i]))
#     plt.plot(LR.iter_steps, l2_train_errors[i], label='L2: Lambda = {}'.format(lambdas[i]))
#     plt.legend(loc='upper right')
#     plt.xlabel('Iterations')
#     plt.ylabel('Percent classified correctly')
#     plt.title('Classification Accuracy\n as a Function of Lambda in Regularization\n'
#                'Across Training')
# plt.show()
#
# for i in xrange(len(lambdas)):
#     plt.plot(LR.iter_steps, l1_train_weights[i], label='L1: Lambda = {}'.format(lambdas[i]))
#     plt.plot(LR.iter_steps, l2_train_weights[i], label='L2: Lambda = {}'.format(lambdas[i]))
#     plt.legend(loc='upper right')
#     plt.xlabel('Iterations')
#     plt.ylabel('Number of Non-zero Weights')
#     plt.title('Number of Non-zero Weights as a Function of Lambda in Regularization')
# plt.show()
#
# plt.plot(np.log10(lambdas), l1_errors, label='L1')
# plt.plot(np.log10(lambdas), l2_errors, label='L2')
# plt.legend(loc='upper right')
# plt.xlabel('Log(Lambda)')
# plt.ylabel('Percent classified correctly')
# plt.title('Classification Accuracy\n as a Function of Lambda in Regularization')
# plt.show()
# for p in xrange(2, 6):
#     lamb = 10 ** p
#
#     LR.initialize_weights()
#     LR.reassign_labels_for_target(2, 3)
#     LR.gradient_descent(1000, log_rate=50, l1=True, lamb=lamb)
#     mnist_printer(LR.weights[1:])
#
#     LR.initialize_weights()
#     LR.reassign_labels_for_target(2, 3)
#     LR.gradient_descent(1000, log_rate=50, l2=True, lamb=lamb)
#     mnist_printer(LR.weights[1:])


# Part 3

SM = SoftMax('mnist', lr_dampener=50)
SM.subset_data(20000, -2000)
SM.assign_holdout(10)

SM.gradient_descent(200, log_rate=10, l2=True, lamb=50000)
SM.plot_logs()
for digit in xrange(SM.weights.shape[1]):
    d = SM.weights[:, digit]
    mnist_printer(d[1:])
avg = np.average(SM.weights[1:, :], axis=1)
mnist_printer(avg)
# SM.initialize_weights()
# SM.gradient_descent(400, log_rate=10, l2=True, lamb=1000)
# SM.plot_logs()
