from Players.Player import Player
import random
import numpy as np
import pickle

class QTPlayer(Player):
    def __init__(self, name, lr=0.2, eps=0.2, gamma=0.97, isLearning=False):
        super().__init__(name)
        self.name = name
        self.lr = lr  # learning rate
        self.eps = eps  # exploration rate
        self.gamma = gamma  # decay rate
        self.isLearning = isLearning
        if not self.isLearning:
            self.eps = 0.0
        self.Qtable = {}
        self.moveHistory = []

    def get_action(self, state):
        actions = state.get_actions()
        if random.random() < self.eps:  # explore
            action = random.choice(actions)
        else:  # exploit
            qs = np.array([self.getQ(state.get_state_hash(), a) for a in actions])
            actionInd = np.random.choice(np.flatnonzero(qs == qs.max()))  # argmax with random tie breaking
            action = actions[actionInd]
        if self.isLearning:
            self.moveHistory.append((state.get_state_hash(), action))
        return action

    def save_policy(self, fileName):
        with open(fileName, 'wb+') as file:
            pickle.dump(self.Qtable, file)

    def load_policy(self, fileName):
        with open(fileName, 'rb') as file:
            self.Qtable = pickle.load(file)

    def learn(self, reward, player):
        if self.isLearning:
            next_q = reward  # termination state gets reward as its value
            for stateHash, action in reversed(self.moveHistory):
                q = self.getQ(stateHash, action)
                self.Qtable[stateHash][action] = q + self.lr * (next_q - q)
                next_q = max(self.Qtable[stateHash].values())
                next_q = self.gamma * next_q

    def getQ(self, state, action):
        if self.Qtable.get(state) is None:
            self.Qtable[state] = {}
        if self.Qtable.get(state).get(action) is None:
            self.Qtable[state][action] = 0.0
        return self.Qtable.get(state).get(action)

    def stop_learning(self):
        self.isLearning = False
        self.eps = 0.0

    def reset(self):
        self.moveHistory = []
        pass

# =================================================================================================================== #

if __name__ == '__main__':
    print("Table Q-learning player")