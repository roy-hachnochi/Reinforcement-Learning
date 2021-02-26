from TicTacToe.TicTacToe import TicTacToeState, tictactoe_dqn_args
from Connect4.Connect4 import Connect4State, connect4_dqn_args
from Players.HumanPlayer import HumanPlayer
from Players.MaxminPlayer import MaxminPlayer
from Players.QTPlayer import QTPlayer
from Players.DQPlayer import DQPlayer
from Players.RandomPlayer import RandomPlayer
from Players.SemiRandomPlayer import SemiRandomPlayer
from Auxiliary import *

def save_if_needed(p, filename):
    if isinstance(p, QTPlayer) or isinstance(p, DQPlayer):
        p.save_policy(filename)

def load_if_needed(p, filename):
    if isinstance(p, QTPlayer) or isinstance(p, DQPlayer):
        p.load_policy(filename)

# =================================================================================================================== #

if __name__ == '__main__':
    mode = "train"  # "train", "evaluate", "play"
    nGames = 100000
    state = TicTacToeState()  # TicTacToeState(), Connect4State()
    policy_fileName = "./TicTacToe/Policies/DQpolicy_selfPlay"  # save/load policy

    # Player options:
    #   HumanPlayer("Roy")
    #   MaxminPlayer("Maxmin", max_depth=5)
    #   QTPlayer("Cutie")
    #   DQPlayer("DQ", tictactoe_dqn_args, isLearning=True)
    #   RandomPlayer("Randy")
    #   SemiRandomPlayer("Randy")
    p1 = DQPlayer("DQ", tictactoe_dqn_args, isLearning=True)
    p2 = SemiRandomPlayer("Randy")

    if mode == "train":
        log = train(state, p1, p2, nGames=nGames)
        plot_log(log)
        save_if_needed(p1, policy_fileName)
        save_if_needed(p2, policy_fileName)

    elif mode == "evaluate":
        load_if_needed(p1, policy_fileName)
        load_if_needed(p2, policy_fileName)
        pWin1, pWin2, pTie = evaluate(state, p1, p2)
        print("{} vs. {}: P1 Win {:.2f}% | P2 Win {:.2f}% | Tie {:.2f}%".format(p1.name, p2.name, pWin1 * 100,
                                                                                pWin2 * 100, pTie * 100))

    elif mode == "play":
        load_if_needed(p1, policy_fileName)
        load_if_needed(p2, policy_fileName)
        game = Game(state, p1, p2)
        game.play()
