import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import pickle
from scipy.optimize import linprog
import pandas as pd
import time
import random
from InvaderDefender import Invader_Defender


# to remove warnings
import warnings
warnings.filterwarnings('ignore')

actions = [[-1, 0], [0, 1], [1, 0], [0, -1]] #up, right, down, left = (clockwise from up) 
action_count = len(actions) 
gridSize = 6 
state_count = gridSize*gridSize

# FUNCTIONS ---------------------------------------------------------------------------
def calculate_value(G_state):
    """
    A function that calculates the value of a game by using linear programming.
    The value is calculated in both the defender and invader's perspective which are equal in value
    and opposite in signs
    Input: payoff matrix of a particular state (4x4 matrix)
    Output: Value = scalar value of the game.
    """
    
    G_state = list(G_state)
    
    # defender lin prog
    c = [0, 0, 0, 0, -1]
    defender_q = -1*np.transpose(G_state)     
    v_coeff = np.ones((4,1))
    Aub = np.concatenate((defender_q,v_coeff),1)
    b = [0, 0, 0, 0]
    Aeq = [[1, 1, 1, 1, 0]]
    beq = [[1.]]
    bounds = ((0,1),(0,1),(0,1),(0,1),(None, None))
    defender_solution = linprog(c, A_ub=Aub, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='revised simplex')
    
    # invader lin prog
    c = [0, 0, 0, 0, 1]
    invader_q = G_state
    w_coeff = np.ones((4,1))*-1
    Aub = np.concatenate((invader_q,w_coeff),1)
    invader_solution = linprog(c, A_ub=Aub, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='revised simplex')
    
    # check if the linprog solution is successful or not
    if defender_solution['status'] == 0:
        value = defender_solution['fun']*-1
    else:
        value = invader_solution['fun']
    
    return value

def calculate_payoff(state, Q):
    """
    A function calculates the payoff of a specific state based on Q values
    Input: state (ie. [0,0,1,1]) and Q dictionary
    Output: payoff = 4x4 matrix where each element represent the defender's payoff 
    when defender take i, and invader take action j
    """
    state = list(state)
    payoff = np.zeros([4,4])
    for i in range(action_count):
        defender_action = i
        for j in range(action_count):
            invader_action = j
            joint_action = [defender_action, invader_action]
            state_action_pair = tuple(state + joint_action)
            payoff[i, j] = Q[state_action_pair]

    return payoff

def equilibrium(G_state):
    """
    A function that obtains the policy for defender and invader
    The value is calculated in both the defender and invader's perspective which are equal in value
    and opposite in signs
    Input: payoff matrix of a particular state (4x4 matrix)
    Output: policy for defender and invader
    """
    
    # defender lin prog
    c = [0, 0, 0, 0, -1]
    defender_q = -1*np.transpose(G_state)     
    v_coeff = np.ones((4,1))
    Aub = np.concatenate((defender_q,v_coeff),1)
    b = [0, 0, 0, 0]
    Aeq = [[1, 1, 1, 1, 0]]
    beq = [[1.]]
    bounds = ((0,1),(0,1),(0,1),(0,1),(None, None))
    defender_solution = linprog(c, A_ub=Aub, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='revised simplex')
    
    # invader lin prog
    c = [0, 0, 0, 0, 1]
    invader_q = G_state
    w_coeff = np.ones((4,1))*-1
    Aub = np.concatenate((invader_q,w_coeff),1)
    invader_solution = linprog(c, A_ub=Aub, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='revised simplex')
    
    if defender_solution['status'] == 0:
        defender_policy = defender_solution['x'][:4]
    else:
        defender_policy = np.array([0.25,0.25,0.25,0.25])
    
    if invader_solution['status'] == 0:
        invader_policy = invader_solution['x'][:4]
    else:
        invader_policy = np.array([0.25,0.25,0.25,0.25])

    return defender_policy, invader_policy

def choose_action(defender_policy, invader_policy, epsilon):
    """
    A function that choose a joint epsilon-greedy action based on defender/invader policy
    Input: defender_policy (1x4), invader policy (1x4), and epsilon (ie. 0.3)
    Output: joint action index = [defender action index, invader action index] = [0 to 3, 0 to 3]
    """
       
    # choose an action type: explore (0) or exploit(1)
    action_type = int(np.random.choice(2, 1, p=[epsilon,1-epsilon]))
    
    if action_type == 0:
        
        # randomly pick an action
        defender_action_index = random.choice(range(4))    
        invader_action_index = random.choice(range(4))    
    
    else:

        # pick best action based on sampling
        defender_action_index = int(np.random.choice(action_count, 1, p=defender_policy.clip(0)))
        invader_action_index = int(np.random.choice(action_count, 1, p=invader_policy.clip(0)))
    
    joint_action = [defender_action_index, invader_action_index]
    
    return joint_action

def generate_trajectory(Defender_state, Invader_state, invader_defender, defender_policy, invader_policy):
    """
    A function that generates a trajectory based on defender and invader's current state
    Input: defender and invader states, invader_defender object, and the policy of defender and invader
    Output: the generated trajectory, the status (who won), the cumulated reward during the trajectory and the total number of steps
    """

    terminal = False
    generated = False
   
    # while not successful generation, repeat
    while not generated:
        game_step = 0
        current_state = tuple(Defender_state + Invader_state)
        game_trajectory = []
        cumulated_reward = 0

        while not terminal:
            
            generated = True    
            
            # append game trajectory
            game_trajectory.append(current_state)

            # check if game is terminal (someone won)
            terminal, status = invader_defender.terminal_check(list(current_state))

            invader_action = actions[int(np.random.choice(action_count, 1, p=invader_policy[tuple(current_state)].clip(0)))]
            defender_action = actions[int(np.random.choice(action_count, 1, p=defender_policy[tuple(current_state)].clip(0)))]

            # obtain next state
            next_state, reward = invader_defender.next_state(list(current_state), defender_action, invader_action)
            current_state = tuple(next_state)
            cumulated_reward = cumulated_reward + reward
            
            game_step += 1
            
            # exit the game if max steps reached (implying stucked)
            if game_step > 200:
                generated = True     
                break
    
    return game_trajectory, status, cumulated_reward, game_step




# ALGORITHM-------------------------------------------------------------------------------

def minimax_qlearning(T, lr, gamma, epsilon, episodes):

    invader_defender = Invader_Defender(6)
    state_list = []
    delta_list = []

    # initialize params
    t = 0
    # T = 200
    # lr = 0.9
    # gamma = 0.95
    # epsilon = 0.99

    # initialize Q matrix
    state_action_pair_list = []

    # create every possible state action pairs: 
    # 1296 states * 4 defender actions * 4 invader actions = 20736 s,a pairs
    for state in invader_defender.game_state_list:
        for defender_action in range(action_count):
            for invader_action in range(action_count):
                joint_action = [defender_action, invader_action]
                state_action_pair = list(state) + joint_action
                state_action_pair_list.append(tuple(state_action_pair))

    # initialize a dictionary for Q values = {(x1, y1, x2, y2, defender_action_index, invader_action_index): q_value}
    listofzeros = [0.0] * len(state_action_pair_list)
    Q = dict(zip(state_action_pair_list, listofzeros))

    # initialize a dictionary for G values = {(x1, y1, x2, y2): payoff_matrix}
    listofzeros = [0.0] * len(invader_defender.game_state_list)
    for state in invader_defender.game_state_list:
        state_list.append(state)
    G = dict(zip(state_list, listofzeros))

    # initialize policies
    initial_policy = []
    defender_policy = {}
    invader_policy = {}
    for i in range(len(invader_defender.game_state_list)):
        random_policy = np.array([0.25, 0.25, 0.25, 0.25])
        initial_policy.append(random_policy)
    defender_policy = dict(zip(state_list, initial_policy))
    invader_policy = dict(zip(state_list, initial_policy))

    # initialize states
    defender_state = [5,0]
    invader_state = [0,0]
    current_state = tuple(defender_state + invader_state)

    # build game based on Q value
    G[current_state] = calculate_payoff(current_state, Q)

    # choose a policy by solving the current game
    defender_policy[current_state], invader_policy[current_state] = equilibrium(G[current_state])

    # initialize a dictionary for G values = {(x1, y1, x2, y2): payoff_matrix}
    listofzeros = [0.0] * len(state_action_pair_list)
    state_count = dict(zip(state_action_pair_list, listofzeros))

    # episodes = 10000
    eps = 0
    delta_list = []

    # testing lists
    testing_reward_list = []
    testing_step_list = []
    testing_game_status = []

    # training lists
    training_step_list = []
    training_reward_list = []

    for eps in range(episodes):
        t = 0
        delta = 0
        
        training_reward = 0
        
        while t < T:
            
            # choose a joint based on epsilon greedy (joint_action = [a1_indx, a2_indx])
            joint_action = choose_action(defender_policy[current_state], invader_policy[current_state], epsilon)
            current_state_action_pair = tuple(list(current_state) + joint_action) # ie. (x1, y1, x2, y2, a1_indx, a2_indx)

            # decaying learning rate
            state_count[current_state_action_pair] += 1
            lr_ = lr / state_count[current_state_action_pair]
    #         lr_ = lr    
        
            # get next state and reward based on current state [x1,y1,x2,y2] and joint action [a1_indx, a2_indx]
            next_state, reward = invader_defender.next_state(current_state, actions[joint_action[0]], actions[joint_action[1]])
            next_state = tuple(next_state)
            training_reward += reward
            
            # build a game based on next state: calculate payoff of next state
            G[next_state] = calculate_payoff(next_state, Q)

            # generate a policy based on equilibirum of next game
            defender_policy[next_state], invader_policy[next_state] = equilibrium(G[next_state])

            # make copy of Q table
            Q_copy = Q[current_state_action_pair]

            # update Q[s,a] <- Q[s,a] + lr*(reward + gamma*value(s') - Q[s,a])
            value = calculate_value(G[next_state])
            Q[current_state_action_pair] = Q[current_state_action_pair] + lr_*(reward + gamma*value - Q[current_state_action_pair])
        
            # if game reached terminal state, restart new episode
            terminal, status = invader_defender.terminal_check(list(next_state))
            if terminal:
                training_steps = t
                defender_state = [5,0]
                invader_state = [0,0]
                current_state = tuple(defender_state + invader_state)
                break

            # calculate delta
            delta = max(delta, abs(Q[current_state_action_pair] - Q_copy))
                    
            # set next state as current state
            current_state = next_state
            t+=1
        
        # Training performance
        training_step_list.append(training_steps)
        training_reward_list.append(training_reward)
        
        # Testing
        game_trajectory, status, test_reward, game_step = generate_trajectory([5,0],[0,0], invader_defender, defender_policy, invader_policy) 
        testing_reward_list.append(test_reward)
        testing_step_list.append(game_step)
        testing_game_status.append(status)
        
        # Print k and current max delta
        clear_output(wait=True)
        display('episode: ' + str(eps) + ' lr: ' + str(lr_) + ' test steps: ' + str(game_step))
        delta_list.append(delta)

    # policy extraction
    for state in invader_defender.game_state_list:    
        G[state] = calculate_payoff(state, Q)
        defender_policy[state], invader_policy[state] = equilibrium(G[state])


    # PLOTTING----------------------------------------------------------------------------
    # Q delta
    plt.plot(delta_list)
    plt.title('Q Delta')
    plt.xlabel('Episode')
    plt.ylabel('Delta')

    # plot moving average
    reward_frame = pd.DataFrame(delta_list)
    rolling_mean = reward_frame.rolling(window=200).mean()
    plt.plot(rolling_mean, label='Moving Average', color='orange')
    plt.legend()
    plt.show()

    # reward per episode
    plt.plot(training_reward_list)
    plt.title('Training Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # plot moving average
    reward_frame = pd.DataFrame(training_reward_list)
    rolling_mean = reward_frame.rolling(window=200).mean()
    plt.plot(rolling_mean, label='Moving Average', color='orange')
    plt.legend()
    plt.show()

    # steps per eipsode
    plt.plot(training_step_list)
    plt.title('Training Game Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    # plot moving average
    reward_frame = pd.DataFrame(training_step_list)
    rolling_mean = reward_frame.rolling(window=200).mean()
    plt.plot(rolling_mean, label='Moving Average', color='orange')
    plt.legend()
    plt.show()

    # reward per episode
    plt.plot(testing_reward_list)
    plt.title('Testing Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # plot moving average
    reward_frame = pd.DataFrame(testing_reward_list)
    rolling_mean = reward_frame.rolling(window=200).mean()
    plt.plot(rolling_mean, label='Moving Average', color='orange')
    plt.legend()
    plt.show()

    # steps per eipsode
    plt.plot(testing_step_list)
    plt.title('Testing Game Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    # plot moving average
    reward_frame = pd.DataFrame(testing_step_list)
    rolling_mean = reward_frame.rolling(window=200).mean()
    plt.plot(rolling_mean, label='Moving Average', color='orange')
    plt.legend()
    plt.show()

    # defender reward
    sum_reward = 0
    sum_reward_list = []
    for i in range(len(testing_reward_list)):
        sum_reward = sum_reward + testing_reward_list[i]
        sum_reward_list.append(sum_reward)
        
    plt.plot(sum_reward_list)
    plt.title('Cumulative Test Reward for Defender')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # invader reward
    sum_reward = 0
    sum_reward_list = []
    for i in range(len(testing_reward_list)):
        sum_reward = sum_reward + (testing_reward_list[i]*-1)
        sum_reward_list.append(sum_reward)
        
    plt.plot(sum_reward_list)
    plt.title('Cumulative Test Reward for Invader')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # GENERATE HEATMAPS ---------------------------------------------------------------------------

    # create Q_state_dict = {(x1,y1,x2,y2): [1,2,3....16]} where 1-16 represent the Q values 
    # of each defender invader action combination
    Q_state_list = []
    for state in invader_defender.game_state_list:
        Q_state = []
        for defender_action in range(action_count):
            for invader_action in range(action_count):
                state_action_pair = tuple(list(state) + [defender_action, invader_action])
                Q_state.append(Q[state_action_pair])
        Q_state_list.append(Q_state)      
        
    Q_state_dict = dict(zip(state_list, Q_state_list))

    # create a list of states that fixes the defender's starting position
    fixed_defender_state_list = []
    for invader_state in invader_defender.states:
        fixed_defender_state = tuple([5, 0] + invader_state)
        fixed_defender_state_list.append(fixed_defender_state)

    invader_map = np.zeros([6,6])
    for state in fixed_defender_state_list:
        invader_map[state[2], state[3]] = max(Q_state_dict[state])*-1  # -1 for invaders perspective

    # if the defender is fixed at the bottom left corner, this heatmap shows the invader's rewards
    plt.imshow(invader_map, interpolation='nearest')
    plt.colorbar()
    plt.title('Q values from the Invader Perspective (Defender fixed at [5,0])')
    plt.show()

    # create a list of states that fixes the invaders's starting position
    fixed_invader_state_list = []
    for defender_state in invader_defender.states:
        fixed_invader_state = tuple(defender_state + [0, 0])
        fixed_invader_state_list.append(fixed_invader_state)

    # create invader heatmap
    defender_map = np.zeros([6,6])
    for state in fixed_invader_state_list:
        defender_map[state[0], state[1]] = max(Q_state_dict[state])

    # if invader is fixed at top left corner, this heatmap shows the defender's rewards
    plt.imshow(defender_map, interpolation='nearest')
    plt.colorbar()
    plt.title('Q values from the Defender Perspective (Invader fixed at [0,0])')
    plt.show()
