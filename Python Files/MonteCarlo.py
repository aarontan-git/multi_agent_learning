import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import uniform
import random
import time
from IPython.display import display, clear_output
from Gridworld import Gridworld
import pickle

# define environment parameters
actions = [[-1, 0], [0, 1], [1, 0], [0, -1]] #up, right, down, left = (clockwise from up) 
action_count = len(actions) # total number of actions
gridSize = 5 # create a square grid of gridSize by gridSize
state_count = gridSize*gridSize # total number of states

def generate_episode(steps, policy):
    """
        A Function generates an episode from a set initial state.
        Input: Number of steps required for an episode
        Output: 3 lists that holds the states visited, action taken and reward observed
    """
    # set initial state
    state_vector = grid.initial_state()

    # initialize state (with initial state), action list and reward list
    state_list = [state_vector]
    action_list = []
    reward_list = []

    # generate an episode
    for i in range(steps):

        # pick an action based on categorical distribution in policy
        action_index = int(np.random.choice(action_count, 1, p=policy[grid.states.index(state_vector)])) 
        action_vector = actions[action_index] # convert the integer index (ie. 0) to action (ie. [-1, 0])

        # get new state and reward after taking action from current state
        new_state_vector, reward = grid.transition_reward(state_vector, action_vector)
        state_vector = list(new_state_vector)

        # save state, action chosen and reward to list
        state_list.append(state_vector)
        action_list.append(action_vector)
        reward_list.append(reward)
        
    return state_list, action_list, reward_list

def Average(lst):
    """
        A Function that averages a list.
        Input: a list
        Output: average value
    """
    return sum(lst) / len(lst) 

# create a grid object
grid = Gridworld(5)

# intialize parameters
gamma = 0.99
epsilon = [0.01, 0.1, 0.25]
runs = 20
episode_length = 500
window_length = int(episode_length/20)
max_steps = 200

# define variables for plotting purposes
reward_epsilon = []
reward_run_all = []
test_reward_epsilon = []
test_reward_run_all = []

# define variables for keeping track of time steps
Terminal = max_steps
t_list=[]
for i in range(max_steps):
    t = Terminal - i - 1
    t_list.append(t)
label = []
for r in range(1, runs+1):
    label.append(str(r))

# Monte Carlo BEGINS ---------------------------------------------------------------------------------------------------------------------------
# begin iterating over every epsilon
for eps in epsilon:

    # reset some lists
    Q_values_list = []
    reward_run = []
    test_reward_run =[]

    # begin iterating over a set amount of runs (20)
    for run in range(1, runs+1):

        # random e soft policy
        policy = np.zeros((state_count, action_count))
        for state in range(len(policy)):
            random_action = random.randint(0,3)
        #     random_action = 0
            for action in range(action_count):
                if action == random_action:
                    policy[state][action] = 1 - eps + eps/action_count 
                else: # if choose_action is not the same as the current action 
                    policy[state][action] = eps/action_count

        # initialize q values for all state action pairs
        Q_values = np.zeros((state_count, action_count))
        oldQ = np.zeros((state_count, action_count))

        # define lists
        reward_episode = []
        test_reward_episode = []
        delta_list = []

        #Modification: added a dictionary of state and list of returns received
        returns_list = {}
        for s in range(state_count):
            for a in range(action_count):
                returns_list[(s,a)] = []

        # iteration 500 times
        for episode in range(episode_length):
        
            # generate an episode of specified step count
            state_list, action_list, reward_list = generate_episode(max_steps, policy)
            # sum reward for episode
            reward_episode.append(sum(reward_list))

            # intialize variables
            G = 0
            delta = 0
            
            # initiate visited list to none
            visited_list = []

            state_action_pair = list(np.zeros(len(t_list)))

            # loop for each step of episode: T-1, T-2, T-3 ... 0 = 199, 198, 197 ... 0
            for t in t_list:

                # calculate G: starting with the last reward at index t (naturally accounts for pseudocode's "t-1")
                G = gamma*G + reward_list[t]
                
                # combine state action pair, for example, state = [0,0], action = [0,1], state_action_pair = [0,0,0,1]
                # state_action_pair = []
                # state_action_pair.extend(state_list[t])
                # state_action_pair.extend(action_list[t])

                state_action_pair[t] = state_list[t]+action_list[t]


                # check if state action pair have been visited before (if not: continue, else: move to the next time step)
                if state_action_pair[t] not in visited_list:

                    # add state action pair to visited list
                    visited_list.append(state_action_pair)
                    
                    # find state and action index, for example, converting action [-1, 0] to 0, and same for state #
                    state_index = grid.states.index(state_list[t])
                    action_index = actions.index(action_list[t])

                    # append G to returns
                    returns_list[(state_index,action_index)].append(G)

                    oldQ[state_index][action_index] = Q_values[state_index][action_index]

                    # write Q_values to the state-action pair
                    Q_values[state_index][action_index] = float(np.mean(returns_list[(state_index,action_index)]))

                    # calculate max delta change for plotting max q value change
                    delta = max(delta, np.abs(Q_values[state_index][action_index] - oldQ[state_index][action_index]))      
            
            #MODIFICATION: adjusted updating rule    
            for s in range(state_count):
                if np.count_nonzero(Q_values[s]) == 0:  # if Q_values is all zero, randomly pick an action
                    choose_action = random.randint(0,3)
                else:
                    choose_action = np.argmax(Q_values[s]) # choose best action at given state
                # overwrite policy
                for a in range(action_count): # for action in actions [0, 1, 2, 3]
                    if choose_action == a: # if the choose_action is the same as the current action
                        policy[s][a] = 1 - eps 
                    else: # if choose_action is not the same as the current action 
                        policy[s][a] = eps/(action_count-1)
            
            # append delta to list
            delta_list.append(delta)
            
            # TESTING AFTER EACH EPISODE ------------------------------------------------------------
            # Generate test trajectory with the greedy policy
            state_list, action_list, test_reward_list = generate_episode(200, policy)
            test_reward_episode.append(sum(test_reward_list))
            #----------------------------------------------------------------------------------------

            # print current episode
            clear_output(wait=True)
            display('Epsilon: ' + str(eps) + ' Run: ' + str(run) + ' Episode: ' + str(episode))
        
        # append lists for plotting purpose
        test_reward_run.append(Average(test_reward_episode))
        reward_run.append(Average(reward_episode))
        Q_values_list.append(Q_values)

        # PLOTTING CODE--------------------------------------------------------------------------------------------------------------------
        # Average Reward per Episode during Training with different runs and epsilons
        # plt.plot(reward_episode)
        # plt.plot(test_reward_episode)
        plt.title('Average Reward per Episode (Smoothed), Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        delta_frame = pd.DataFrame(test_reward_episode)
        rolling_mean = delta_frame.rolling(window=window_length).mean()
        plt.plot(rolling_mean, label='Moving Average')
        delta_frame = pd.DataFrame(reward_episode)
        rolling_mean = delta_frame.rolling(window=window_length).mean()
        plt.plot(rolling_mean, label='Moving Average')
        plt.legend(('Testing','Training'))
        plt.savefig('Graphs/MonteCarlo/reward_episode/reward_episode_smoothed_run_' + str(int(run)) + '_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.05)

        # Average Reward per Episode during Training with different runs and epsilons
        plt.plot(test_reward_episode)
        plt.plot(reward_episode)
        plt.title('Average Reward per Episode, Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        # delta_frame = pd.DataFrame(reward_episode)
        # rolling_mean = delta_frame.rolling(window=window_length).mean()
        # plt.plot(rolling_mean, label='Moving Average', color='blue')
        # delta_frame = pd.DataFrame(test_reward_episode)
        # rolling_mean = delta_frame.rolling(window=window_length).mean()
        # plt.plot(rolling_mean, label='Moving Average', color='orange')
        plt.legend(('Testing','Training'))
        plt.savefig('Graphs/MonteCarlo/reward_episode/reward_episode_run_' + str(int(run)) + '_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.05)

        # max delta of each episode, where delta is the change in Q values
        plt.plot(delta_list)
        plt.title('Monte Carlo Max Delta for Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
        plt.xlabel('Episode')
        plt.ylabel('Max Delta')
        # plot moving average
        delta_frame = pd.DataFrame(delta_list)
        rolling_mean = delta_frame.rolling(window=window_length).mean()
        plt.plot(rolling_mean, label='Moving Average', color='orange')
        plt.savefig('Graphs/MonteCarlo/delta/delta_run_'+str(int(run))+'_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.05)

    # append lists for plotting
    reward_run_all.append(reward_run)
    test_reward_run_all.append(test_reward_run)
    reward_epsilon.append(Average(reward_run))
    test_reward_epsilon.append(Average(test_reward_run))

    # Average Reward for each Run with different Epsilon
    plt.plot(test_reward_run)
    plt.plot(reward_run)
    plt.title('Average Reward for each Run with Epsilon: '+ str(float(eps)))
    plt.xlabel('Run')
    plt.xticks(np.arange(runs), label)
    plt.ylabel('Average Reward')
    plt.legend(('Testing','Training'))
    plt.savefig('Graphs/MonteCarlo/reward_run/reward_run_epsilon_' + str(float(eps)) + '.png')
    plt.clf()
    time.sleep(0.05)

    # save Q value tables to a pickle
    with open('Graphs/MonteCarlo/Qvalues/MonteCarlo_Qvalues_' + str(eps) + '.pkl', 'wb') as f:
        pickle.dump(Q_values_list, f)

# Average Reward for each Epsilon
x_label = ('0.01', '0.1', '0.25')
plt.bar(x_label, reward_epsilon)
# plt.plot(reward_epsilon)
plt.title('Average Reward for each Epsilon during Training')
plt.xlabel('Epsilon')
plt.xticks(np.arange(3), ('0.01', '0.1', '0.25'))
plt.ylabel('Average Reward')
plt.savefig('Graphs/MonteCarlo/reward_epsilon/reward_epsilon.png')
plt.clf()
time.sleep(0.05)

# Average Reward for Each Epsilon
x_label = ('0.01', '0.1', '0.25')
plt.bar(x_label, test_reward_epsilon)
# plt.plot(test_reward_epsilon)
plt.title('Average Reward for Each Epsilon during Testing')
plt.xlabel('Epsilon')
plt.xticks(np.arange(3), ('0.01', '0.1', '0.25'))
plt.ylabel('Average Reward')
plt.savefig('Graphs/MonteCarlo/test_reward_epsilon/test_reward_epsilon.png')
plt.clf()
time.sleep(0.05)

# Average Reward for each Run during Training
for r in range(3):
    plt.plot(reward_run_all[r])
plt.title('Average Reward for each Run during Training')
plt.xlabel('Run')
plt.xticks(np.arange(runs), label)
plt.ylabel('Average Reward')
plt.legend(('0.01','0.1','0.25'))
plt.savefig('Graphs/MonteCarlo/reward_run/reward_run_all.png')
plt.clf()
time.sleep(0.05)

# Average Reward for each Run during Testing
for r in range(3):
    plt.plot(test_reward_run_all[r])
plt.title('Average Reward for each Run during Testing')
plt.xlabel('Run')
plt.xticks(np.arange(runs), label)
plt.ylabel('Average Reward')
plt.legend(('0.01','0.1','0.25'))
plt.savefig('Graphs/MonteCarlo/test_reward_run/test_reward_run_all.png')
plt.clf()
time.sleep(0.05)