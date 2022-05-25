import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=1.0, gamma=0.7, eps_start=0.001, eps_decay=.99999, eps_min=0.00001):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = eps_start
        self.alpha = alpha
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        

    def epsilon_greedy_probs(self, Q_s, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        policy_s = self.epsilon_greedy_probs(self.Q[state], self.epsilon)
        # pick action A
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            self.Q[state][action] = self.Q[state][action] + self.alpha*( reward + self.gamma*(np.max(self.Q[next_state])) 
                                                         - self.Q[state][action])
        else:
            self.Q[state][action] = self.Q[state][action] + self.alpha*( reward - self.Q[state][action])
            
            self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
            #self.eps = max(self.eps**self.i_episode, self.eps_min)