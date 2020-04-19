import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import pickle
from scipy.optimize import linprog
import pandas as pd
import time
from InvaderDefender import Invader_Defender

# to remove warnings
import warnings
warnings.filterwarnings('ignore')

actions = [[-1, 0], [0, 1], [1, 0], [0, -1]] #up, right, down, left = (clockwise from up) 
action_count = len(actions) 
gridSize = 6 
state_count = gridSize*gridSize

# Functions -----------------------------------------------
def calculate_payoff(state, invader_defender, gamma, U, k):
    """
    A function calculates the payoff of a specific state by iterating over every defender/invader action
    Input: state (ie. [0,0,1,1]), invader_defender object, gamma, U and k
    Output: payoff = 4x4 matrix where each element represent the defender's payoff 
    when defender take i, and invader take action j
    """
    state = list(state)
    payoff = np.zeros([4,4])
    for i in range(action_count):
        defender_action = actions[i]
        for j in range(action_count):
            invader_action = actions[j]
            next_state, reward = invader_defender.next_state(state, defender_action, invader_action)
            payoff[i, j] = reward + gamma*invader_defender.transition_prob*U[k][tuple(next_state)]
    return payoff

def calculate_value(G_state):
    """
    A function that calculates the value of a game by using linear programming.
    The value is calculated in both the defender and invader's perspective which are equal in value
    and opposite in signs
    Input: payoff matrix of a particular state (4x4 matrix)
    Output: Value = scalar value of the game.
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
    
    # check if the linprog solution is successful or not
    if defender_solution['status'] == 0:
        value = defender_solution['fun']*-1
    else:
        value = invader_solution['fun'] 
    
    return value

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



# SHAPLEYS VALUE ITERATION---------------------------------------------------
def shapley_value_iteration(tolerance, gamma):

    # create invader defender object
    invader_defender = Invader_Defender(6)

    # initialize parameters
    k = 0
    U = {}
    state_list = []
    listofzeros = [0.0] * len(invader_defender.game_state_list)
    delta_list = []

    # convert game_state_list in to a state list of tuples in order to make a dictionary
    for state in invader_defender.game_state_list:
        state_list.append(state)
        
    # initiate params
    G = dict(zip(state_list, listofzeros))
    U[k] = dict(zip(state_list, listofzeros))

    # initialize policies
    initial_policy = []
    for i in range(len(invader_defender.game_state_list)):
        random_policy = np.array([0.25, 0.25, 0.25, 0.25])
        initial_policy.append(random_policy)
    defender_policy = dict(zip(state_list, initial_policy))
    invader_policy = dict(zip(state_list, initial_policy))

    # to start the loop
    delta = tolerance + 1
    k = 0

    last_delta = 1000

    # START THE VALUE ITERATION ALGORITHM
    while delta > tolerance:
        delta = 0
        
        # initialize the next entry of the U dictionary
        U[k+1] = dict(zip(state_list, listofzeros))

        for state in invader_defender.game_state_list:
            
            # Build G dictionary {state: payoff (4x4)}
            G[state] = calculate_payoff(state, invader_defender, gamma, U, k)

            # calculate value of game
            value = calculate_value(G[state])

            # write value of game to the dictionary
            U[k+1][state] = value

            # calculate delta
            delta = max(delta, abs(U[k+1][state]-U[k][state]))
            
        # print k and current max delta
        # clear_output(wait=True)
        print('k: ' + str(k) + ' delta: ' + str(delta))
        
        delta_list.append(delta)
        k += 1
        
        # stop training when delta explodes
        if delta > last_delta:
            break
        
        last_delta = delta

    # policy extraction
    for state in invader_defender.game_state_list:  
        G[state] = calculate_payoff(state, invader_defender, gamma, U, k)
        defender_policy[state], invader_policy[state] = equilibrium(G[state])


    # PLOTTING -----------------------------------------------------------------------------------------

    # plot delta
    plt.plot(delta_list)
    plt.title('Iteration vs Delta')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.show()

    # GENERATE HEATMAPS ------------------------------------------------------------------

    # create a list of states that fixes the defender's starting position
    fixed_defender_state_list = []
    for invader_state in invader_defender.states:
        fixed_defender_state = tuple([5, 0] + invader_state)
        fixed_defender_state_list.append(fixed_defender_state)

    # create invader heatmap
    invader_map = np.zeros([6,6])
    for state in fixed_defender_state_list:
        invader_map[state[2], state[3]] = U[k][state]*-1 # -1 for invaders perspective

    # if the defender is fixed at the bottom left corner, this heatmap shows the invader's rewards
    plt.imshow(invader_map, interpolation='nearest')
    plt.colorbar()
    plt.title('Value Function from the Invader Perspective (Defender fixed at [5,0])')
    plt.show()

    # create a list of states that fixes the invaders's starting position
    fixed_invader_state_list = []
    for defender_state in invader_defender.states:
        fixed_invader_state = tuple(defender_state + [0, 0])
        fixed_invader_state_list.append(fixed_invader_state)

    # create invader heatmap
    defender_map = np.zeros([6,6])
    for state in fixed_invader_state_list:
        defender_map[state[0], state[1]] = U[k][state]

    # if invader is fixed at top left corner, this heatmap shows the defender's rewards
    plt.imshow(defender_map, interpolation='nearest')
    plt.colorbar()
    plt.title('Value Function from the Defender Perspective (Invader fixed at [0,0])')
    plt.show()