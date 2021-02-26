import numpy as np
from Players.DQPlayer import DQNArgs
from Utility.Consts import Consts

BOARD_DIM = 3
N_CH = 2
P1S = 'x'
P2S = 'o'

class TicTacToeState:
    def __init__(self):
        self.name = "Tic Tac Toe"
        self.board = np.zeros((BOARD_DIM, BOARD_DIM))
        self.lastAction = []
        self.curPlayer = Consts.P1

    def get_state(self):
        return self.board

    def get_board(self):
        otherPlayer = Consts.P2 if self.curPlayer == Consts.P1 else Consts.P1
        p1_board = 1 * (self.board == self.curPlayer)
        p2_board = 1 * (self.board == otherPlayer)
        return np.stack((p1_board, p2_board), axis=0)

    def get_state_hash(self):
        return str(self.board.reshape(-1))

    def show_state(self):
        s = ''.join(['-'] + ['-' for _ in range(4*BOARD_DIM)])
        print(s)
        for row in range(BOARD_DIM):
            rowStr = '| '
            for col in range(BOARD_DIM):
                if self.board[row, col] == Consts.P1:
                    rowStr = rowStr + P1S + ' | '
                elif self.board[row, col] == Consts.P2:
                    rowStr = rowStr + P2S + ' | '
                else:
                    rowStr = rowStr + '  | '
            rowStr = rowStr[:-1]  # remove last space
            print(rowStr)
        print(s)

    def get_all_actions(self):
        actions = []
        for row in range(BOARD_DIM):
            for col in range(BOARD_DIM):
                actions.append((row, col))
        return actions

    def get_actions(self):
        all_actions = self.get_all_actions()
        actions = [action for action in all_actions if self.check_legal(action)]
        return actions

    def get_illegal_actions(self):
        all_actions = self.get_all_actions()
        illegal_inds = [not self.check_legal(action) for action in all_actions]
        return illegal_inds

    def check_legal(self, action):
        if action[0] < 0 or action[0] >= BOARD_DIM or action[1] < 0 or action[1] >= BOARD_DIM:
            return False
        return self.board[action] == 0

    def get_action_command(self):
        return 'Enter desired cell as [row col]:\n'

    def play(self, action):
        action = tuple([int(action[0]), int(action[1])])
        if not self.check_legal(action):  # illegal action
            return False, False
        self.board[action] = self.curPlayer
        self.curPlayer = Consts.P2 if self.curPlayer == Consts.P1 else Consts.P1
        self.lastAction.append(action)
        isEnd, _ = self.check_win()
        return True, isEnd

    def check_win(self):
        # We need to check only row/col/diags containing last move
        if not self.lastAction:
            return False, 0
        action = self.lastAction[-1]
        row, col = action
        colSum = np.sum(self.board[row, :])
        rowSum = np.sum(self.board[:, col])
        allSums = [colSum, rowSum]
        if row == col:  # check diagonal
            diagSum1 = np.trace(self.board)
            allSums = allSums + [diagSum1]
        if row + col == BOARD_DIM - 1:  # check anti-diagonal
            diagSum2 = np.trace(np.fliplr(self.board))
            allSums = allSums + [diagSum2]
        if BOARD_DIM*Consts.P1 in allSums:  # player 1 wins
            return True, Consts.P1
        elif BOARD_DIM*Consts.P2 in allSums:  # player 2 wins
            return True, Consts.P2
        elif np.all((self.board != 0)):  # no more moves - tie
            return True, Consts.TIE
        else:  # continue playing
            return False, 0

    def get_rewards(self):
        isEnd, winner = self.check_win()
        if isEnd:
            if winner == Consts.P1:
                return 1, -1
            if winner == Consts.P2:
                return -1, 1
            if winner == Consts.TIE:
                return 0.2, 0.5  # player 1 started so has an advantage - punish him more for a tie
        return 0, 0

    def get_heuristic(self, player):
        score = 0
        for i in range(BOARD_DIM):
            score += line_heuristic(self.board[i, :], player)  # row
            score += line_heuristic(self.board[:, i], player)  # col
        score += line_heuristic(np.diag(self.board), player)  # main diagonal
        score += line_heuristic(np.diag(np.fliplr(self.board)), player)  # anti-diagonal
        return score

    def undo(self):
        if self.lastAction:
            self.board[self.lastAction[-1]] = 0
            self.lastAction = self.lastAction[:-1]
            self.curPlayer = Consts.P2 if self.curPlayer == Consts.P1 else Consts.P1

    def reset(self):
        self.board = np.zeros((BOARD_DIM, BOARD_DIM))
        self.lastAction = []
        self.curPlayer = Consts.P1


tictactoe_dqn_args = DQNArgs(ch=2,
                             h=BOARD_DIM,
                             w=BOARD_DIM,
                             output_size=BOARD_DIM**2,
                             layer_channels=[32, 32],
                             layer_sizes=[3, 3],
                             layer_strides=[1, 1],
                             layer_padding=[1, 1],
                             batch_size=32,
                             mem_size=100000,
                             target_update=5000,
                             eps_decay=2e4,
                             lr=0.001,
                             gamma=0.99)


def line_heuristic(l, player):
    score_factor = 10
    score = 0
    for i in range(len(l)):
        if l[i] == player:  # players piece
            if score == 0:  # seen nothing up to here
                score = 1
            elif score > 0:  # seen up to here only players pieces
                score *= score_factor
            else:  # seen opponents pieces -> line has no value
                return 0
        elif l[i] == 0:  # empty
            continue
        else:  # opponents piece
            if score == 0:  # seen nothing up to here
                score = -1
            elif score < 0:  # seen up to here only opponents pieces
                score *= score_factor
            else:  # seen players pieces -> line has no value
                return 0
    return score

# =================================================================================================================== #

if __name__ == '__main__':
    print("Tic Tac Toe")
