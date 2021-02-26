import numpy as np
from Players.DQPlayer import DQNArgs
from Utility.Consts import Consts

N_ROWS = 6
N_COLS = 7
N_WIN = 4
N_CH = 2
P1S = 'x'
P2S = 'o'

class Connect4State:
    def __init__(self):
        self.name = "Connect" + str(N_WIN)
        self.board = np.zeros((N_ROWS, N_COLS))
        self.lowest = [N_ROWS - 1] * N_COLS
        self.lastAction = []
        self.curPlayer = Consts.P1

    def get_state(self):
        return self.board

    def get_board(self):
        p1_board = 1 * (self.board == Consts.P1)
        p2_board = 1 * (self.board == Consts.P2)
        return np.dstack((p1_board, p2_board)).transpose([2, 1, 0])

    def get_state_hash(self):
        return str(self.board.reshape(-1))

    def show_state(self):
        s = ''.join(['-'] + ['-' for _ in range(4*N_COLS)])
        print(s)
        for row in range(N_ROWS):
            rowStr = '| '
            for col in range(N_COLS):
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
        actions = [tuple([col]) for col in range(N_COLS)]
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
        if action[0] < 0 or action[0] >= N_COLS:
            return False
        return self.lowest[action[0]] >= 0

    def get_action_command(self):
        return 'Enter desired column (from 0):\n'

    def play(self, action):
        action = tuple([int(action[0])])
        if not self.check_legal(action):  # illegal action
            return False, False
        action = action[0]
        self.board[self.lowest[action], action] = self.curPlayer
        self.lowest[action] -= 1
        self.curPlayer = Consts.P2 if self.curPlayer == Consts.P1 else Consts.P1
        self.lastAction.append(action)
        isEnd, _ = self.check_win()
        return True, isEnd

    def check_win(self):
        # We need to check only rows/cols/diags containing last move
        if not self.lastAction:
            return False, 0
        action = self.lastAction[-1]
        row, col = self.lowest[action] + 1, action

        # check row:
        result = check_line_win(list(self.board[row, :]))
        if result != 0:
            winner = Consts.P1 if result == Consts.P1 else Consts.P2
            return True, winner

        # check col:
        result = check_line_win(list(self.board[:, col]))
        if result != 0:
            winner = Consts.P1 if result == Consts.P1 else Consts.P2
            return True, winner

        # check diagonal 1:
        lims = [-min(row, col), min(N_ROWS - row, N_COLS - col)]
        result = check_line_win([self.board[row + i, col + i] for i in range(lims[0], lims[1])])
        if result != 0:
            winner = Consts.P1 if result == Consts.P1 else Consts.P2
            return True, winner

        # check diagonal 2:
        lims = [-min(row, N_COLS - col - 1), min(N_ROWS - row, col + 1)]
        result = check_line_win([self.board[row + i, col - i] for i in range(lims[0], lims[1])])
        if result != 0:
            winner = Consts.P1 if result == Consts.P1 else Consts.P2
            return True, winner

        # check tie:
        if np.all(self.board != 0):
            return True, Consts.TIE
        return False, 0

    def get_rewards(self):
        isEnd, winner = self.check_win()
        if isEnd:
            if winner == Consts.P1:
                return 1, -1
            if winner == Consts.P2:
                return -1, 1
            if winner == Consts.TIE:
                return 0.3, 0.3
        return 0, 0

    def get_heuristic(self, player):
        score = 0

        # add row scores:
        for row in range(N_ROWS):
            for col in range(N_COLS - N_WIN + 1):
                score += window_heuristic(self.board[row, col:(col + N_WIN)], player)

        # add column scores:
        for col in range(N_COLS):
            for row in range(N_ROWS - N_WIN + 1):
                score += window_heuristic(self.board[row:(row + N_WIN), col], player)

        # add diagonal 1 scores:
        for row in range(N_ROWS - N_WIN + 1):
            for col in range(N_COLS - N_WIN + 1):
                window = np.array([self.board[row + i, col + i] for i in range(N_WIN)])
                score += window_heuristic(window, player)

        # add diagonal 2 scores:
        for row in range(N_ROWS - N_WIN + 1):
            for col in range(N_WIN - 1, N_COLS):
                window = np.array([self.board[row + i, col - i] for i in range(N_WIN)])
                score += window_heuristic(window, player)

        return score

    def undo(self):
        if self.lastAction:
            action = self.lastAction[-1]
            self.board[self.lowest[action] + 1, action] = 0
            self.lowest[action] += 1
            self.lastAction = self.lastAction[:-1]
            self.curPlayer = Consts.P2 if self.curPlayer == Consts.P1 else Consts.P1

    def reset(self):
        self.board = np.zeros((N_ROWS, N_COLS))
        self.lowest = [N_ROWS - 1] * N_COLS
        self.lastAction = []
        self.curPlayer = Consts.P1


connect4_dqn_args = DQNArgs(ch=N_CH,
                            h=N_ROWS,
                            w=N_COLS,
                            output_size=N_COLS,
                            layer_channels=[16, 32, 32, 16],
                            layer_sizes=[3, 3, 3, 3],
                            layer_strides=[1, 1, 2, 2],
                            layer_padding=[1, 1, 0, 0],
                            batch_size=32,
                            mem_size=100000,
                            target_update=5000,
                            eps_decay=2e4,
                            lr=0.001,
                            gamma=0.99)


def check_line_win(l):
    # Checks if there exists a winning series (N_WIN of the same tile in a line).
    for i in range(0, len(l) - N_WIN + 1):
        cur_l = l[i:(i + N_WIN)]  # current values to check
        if len(set(cur_l)) == 1 and cur_l[0] != 0:  # if all values are equal to P1 or P2 - found winner
            return cur_l[0]
    return 0

def window_heuristic(window, player):
    score_factor = 10
    num_player = np.sum(window == player)
    num_empty = np.sum(window == 0)
    num_opp = len(window) - num_player - num_empty
    if num_player > 0 and num_opp > 0:  # no player can get a win
        return 0
    if num_player > 0:
        return score_factor ** num_player
    if num_opp > 0:
        return -1 * (score_factor ** num_opp)
    return 0  # empty line

# =================================================================================================================== #

if __name__ == '__main__':
    print("Connect 4")
