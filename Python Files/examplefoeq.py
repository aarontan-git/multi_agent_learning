# FINAL
import pickle
from environment_st import *
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
np.random.seed(4321)
ac_def = { '0': 'up', '1': 'down', '2': 'right', '3': 'left', '4': 'stick' }

class Agent():
    def __init__(self, S, A):
        self.Q  = np.zeros((S, A, A))

def FOEQ(Q):
  actions = 5
  # (5) Row robability constraints
  row_A = np.array(-Q, dtype="float") # -1 to minimize
  row_b = np.zeros(len(row_A), dtype="float")
  # (5) Probability constraint
  prob_A = -np.identity(actions, dtype="float")
  prob_b = np.zeros(actions, dtype="float")
  # (2) Equality constraints Sum(prob) = 1
  equality_A = np.array([np.ones((actions)), -np.ones((actions))], dtype="float")
  equality_b = np.array([1, -1], dtype="float")
  #  Auxiliary variable Z
  z = np.array([1.0 for i in range(actions)] + [0.0 for i in range(actions + 2)])
  A = matrix(np.hstack((np.vstack((np.transpose(row_A), prob_A, equality_A)), z.reshape(z.size, 1))))
  b = matrix(np.concatenate((row_b, prob_b, equality_b)))
  c = matrix(np.hstack(([0.0 for i in range(actions)], [-1.0])))
  solvers.options['glpk'] = {'msg_lev' : 'GLP_MSG_OFF'}
  solution = solvers.lp(c, A, b, solver='glpk')
  if (solution['x'] != None):
    value = -solution['primal objective'] # Max Z = Min -Z => We need Z
    #pi    = np.array([sol if sol > 0 else 0 for sol in solution['x'][0:5]])
    #pi    = pi/np.sum(pi)
  else:
    value = 0
    #pi    = [0.2 for i in range(5)]
  #action = action = np.random.choice([0,1,2,3,4], p = pi)
  return value #action, value

