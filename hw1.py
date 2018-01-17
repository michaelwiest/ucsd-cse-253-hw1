from lr import *
from sm import *
from helper import *


# Part 1

RL = LinearRegressor('mnist', lr_dampener=75)
RL.subset_data(20000, -2000)

RL.assign_holdout(10)
RL.reassign_labels_for_target(2, 3)
RL.gradient_descent(2000, log_rate=50) #, l1=True, lamb=1000)
RL.plot_logs()
w = RL.weights[1:]


RL.initialize_weights()
RL.reassign_labels_for_target(2, 8)
RL.gradient_descent(2000, log_rate=50) #, l1=True, lamb=1000)
RL.plot_logs()
w2 = RL.weights[1:]
mnist_printer(w)
mnist_printer(w2)
mnist_printer(w-w2)

# SM = SoftMax('mnist', lr_dampener=50)
# SM.subset_data(10000, -200)
# SM.assign_holdout(10)
#
# SM.gradient_descent(100, log_rate=10, l1=True, lamb=1000)
# SM.plot_logs()
# SM.initialize_weights()
# SM.gradient_descent(400, log_rate=10, l2=True, lamb=1000)
# SM.plot_logs()
