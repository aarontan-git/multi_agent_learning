import numpy as np

class Invader_Defender():
    def __init__(self, gridSize):
        self.valueMap = np.zeros((gridSize, gridSize))
        self.states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
        self.size = gridSize
        
        # deterministic transition
        self.transition_prob = 1 
        
        # initialize defender and invader states
        self.new_state = [0, 0, 0, 0]
        self.new_defender_state = [0, 0]
        self.new_invader_state = [0, 0]
        
        # set territory state
        self.territory_state = [4, 4]

        # create a list of all possible states in the game
        self.game_state_list = []
        for defender_state in self.states:
            for invader_state in self.states:
                combined_states = tuple(defender_state + invader_state)
                self.game_state_list.append(combined_states)
        
        # create 2 lists of states representing defender and invader victory
        self.defender_won = []
        self.invader_won = []
        
        # create states representing defender victory
        for defender_state in self.states:
            for invader_state in self.states:
                distance = np.linalg.norm(np.array(defender_state) - np.array(invader_state))
                # if the invader is not at territory and within the capture range of defender = defender won
                if invader_state != self.territory_state and distance <= np.sqrt(2):
                    combined_states = defender_state + invader_state
                    self.defender_won.append(combined_states)
           
        # create states representing invader victory = anytime invader is at territory
        for defender_state in self.states:               
            combined_states = defender_state + self.territory_state
            self.invader_won.append(combined_states)
    
    def possible_states(self):
        """
        A function that returns a list of all possible states in the game
        """
        return self.game_state_list
    
    def terminal_check(self, state):
        """
        A function that checks whether the game is at a terminal state.
        Terminal state happens when either the invader or defender has won.
        """
        if state in self.defender_won:
            status = "Defender Won"
            terminal_check = True
        elif state in self.invader_won:
            status = "Invader Won"
            terminal_check = True
        else:
            terminal_check = False
            status = "Game in Progress"

        return terminal_check, status
    
    def transition_probability(self, transition):
        """
        A function that returns the transition probability.
        For convenience, the calculation of the reward is moved to the next function so that
        a single function can return the next state as well as the reward at the same time.
        """
        return self.transition_prob

    def next_state(self, current_state, defender_action, invader_action):
        """
        A function that returns the next state
        Input: current state [0,0] , defender_action [0, 1], invader_action [0,-1]
        Output: next state array([x1,y1,x2,y2]) and reward (int)
            - If the action takes the agent off grid, the agent remains in original state
            - If defender won, reward is calculated based on manhattan distance between invader captured state
            and territory
            - If defender loss, reward is -100
        """
        defender_state = []
        invader_state = []
        
        # deconstruct current state [0,0,1,1] in to defender [0,0] and invader [1,1] state
        for i in range(4):
            if i < 2:
                defender_state.append(current_state[i])
            else:
                invader_state.append(current_state[i])
                
        # get next state: state: [0, 0], action: [0, 1], new_state = [0, 1]
        self.new_defender_state = list(np.array(defender_state) + np.array(defender_action))
        self.new_invader_state = list(np.array(invader_state) + np.array(invader_action))

        # if new defender states results in off the grid, return to original state
        if -1 in self.new_defender_state or self.size in self.new_defender_state:
            self.new_defender_state = defender_state
        
        # if new invader states results in off the grid, return to original state
        if -1 in self.new_invader_state or self.size in self.new_invader_state:
            self.new_invader_state = invader_state
       
        # combine the defender and invader state
        self.new_state = self.new_defender_state
        self.new_state.extend(self.new_invader_state)
        
        # reward calculation
        terminal, status = self.terminal_check(self.new_state)
        if terminal == True:
            if status == "Defender Won":
                # defender reward if defender won (manhattan distance between invader captured state and territory)
                distance_to_territory = sum(abs(np.array(self.new_invader_state) - np.array(self.territory_state)))
                self.reward = distance_to_territory * 10
            else:
                # defender reward if invader won
                self.reward = -1000
        else:
            # penalize defender for every step that invader takes closer to territory
            invader_to_territory = sum(abs(np.array(self.new_invader_state) - np.array(self.territory_state)))
            # '8' is the max distance from the invader's initial position to the territory
            self.reward = -(8 - invader_to_territory)*10
            
        return self.new_state, self.reward