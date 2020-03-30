# import Q Learning function
from QLearning1 import QLearning

# define experiment parameters
gamma = 0.99
lr = 0.1
epsilon = [0.01, 0.1, 0.25]
runs = 2
step_number = 100
episode_length = 100

# run experiment
QLearning(gamma, lr, epsilon, runs, step_number, episode_length)