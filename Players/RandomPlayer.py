from Players.Player import Player
import random

class RandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def get_action(self, state):
        actions = state.get_actions()
        return random.choice(actions)

# =================================================================================================================== #

if __name__ == '__main__':
    print("Random player")

