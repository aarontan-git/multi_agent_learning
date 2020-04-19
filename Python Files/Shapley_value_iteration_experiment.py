# import shapley value iteration function
from Shapley_value_iteration import shapley_value_iteration

# define experiment parameters
tolerance = 1e-6
gamma = 0.95

# start experiment
shapley_value_iteration(tolerance, gamma)
