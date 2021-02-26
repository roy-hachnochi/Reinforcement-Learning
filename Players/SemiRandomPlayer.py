from Players.Player import Player
import copy
from Utility.Consts import Consts
import random

class SemiRandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.player = Consts.P1

    def get_action(self, state):
        statec = copy.copy(state)
        self.__set_player(state.curPlayer)
        actions = statec.get_actions()
        for action in actions:
            statec.play(action)
            isEnd, winner = statec.check_win()
            statec.undo()
            if isEnd and winner == state.curPlayer:
                return action
        return random.choice(actions)

    def __set_player(self, p):
        self.player = p

# =================================================================================================================== #

if __name__ == '__main__':
    print("Semi-Random player")

