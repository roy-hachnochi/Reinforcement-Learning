from Players.Player import Player
import copy
import random
from Utility.Consts import Consts
import numpy as np

class MaxminPlayer(Player):
    def __init__(self, name, max_depth=4):
        super().__init__(name)
        self.player = Consts.P1
        self.max_depth = max_depth

    def get_action(self, state):
        statec = copy.copy(state)
        self.__set_player(state.curPlayer)
        actions = statec.get_actions()
        scores = []
        for action in actions:
            statec.play(action)
            scores.append(self.__maxmin(statec, False, 0, -np.inf, np.inf))
            statec.undo()
        best_score = max(scores)
        best_actions = [actions[i] for i in range(len(actions)) if scores[i] == best_score]
        return random.choice(best_actions)

    def __set_player(self, p):
        self.player = p

    def __maxmin(self, state, isMaxPlayer, depth, alpha, beta):
        isLeaf, _ = state.check_win()
        if isLeaf or depth >= self.max_depth:
            score = state.get_heuristic(self.player)
            return score

        bestR = -np.inf if isMaxPlayer else np.inf
        actions = state.get_actions()
        for action in actions:
            state.play(action)
            r = self.__maxmin(state, not isMaxPlayer, depth + 1, alpha, beta)
            state.undo()
            if isMaxPlayer:
                bestR = max(bestR, r)
                alpha = max(alpha, bestR)
            else:
                bestR = min(bestR, r)
                beta = min(beta, bestR)
            if beta <= alpha:
                break
        return bestR

    def reset(self):
        self.player = Consts.P1

# =================================================================================================================== #

if __name__ == '__main__':
    print("Minimax Player")