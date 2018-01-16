from lr import *
from sm import *
from helper import *

RL = LinearRegressor('mnist', lr_dampener=75)
RL.subset_data(20000, -200)
RL.assign_holdout(10)

RL.reassign_labels_for_target(2)
RL.gradient_descent(500, log_rate=50)
RL.plot_logs()



SM = SoftMax('mnist', lr_dampener=50)
SM.subset_data(20000, -200)
SM.assign_holdout(10)

SM.gradient_descent(400, log_rate=10)
SM.plot_logs()
