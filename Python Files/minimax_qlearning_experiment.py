# import minimax q learning function
from minimax_qlearning import minimax_qlearning

# define experiment parameters
Max_episode_length = 200
learning_rate = 0.9
gamma = 0.95
epsilon = 0.99
episodes = 10000

# start experiment
minimax_qlearning(Max_episode_length, learning_rate, gamma, epsilon, episodes)