from Utility.Game import Game
from Utility.Consts import Consts
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# =================================================================================================================== #
def train(state, player1, player2=None, nGames=100000):
    is_self_play = player2 is None
    if is_self_play:
        player2 = copy.deepcopy(player1)
        player2.stop_learning()
    nUpdate = 20000  # for self play - update player 2 every nUpdate episodes
    nLog = 1000  # save log every nLog episodes
    winCount = {Consts.TIE: 0, Consts.P1: 0, Consts.P2: 0}
    log = {Consts.TIE: [], Consts.P1: [], Consts.P2: [], 'episode': []}
    for i in tqdm(range(nGames)):
        # switch players to let them learn from both types:
        if i % 2:
            game = Game(state, player1, player2)
        else:
            game = Game(state, player2, player1)

        # play:
        r1, r2, winner = game.play()

        # log:
        if winner != Consts.TIE and not i % 2:  # players switched order - switch back winner
            winner = Consts.P1 if winner == Consts.P2 else Consts.P2
        winCount[winner] += 1
        if i % nLog == nLog - 1:
            log[Consts.TIE].append(100 * winCount[Consts.TIE] / nLog)
            log[Consts.P1].append(100 * winCount[Consts.P1] / nLog)
            log[Consts.P2].append(100 * winCount[Consts.P2] / nLog)
            log['episode'].append(i + 1)
            winCount = {Consts.TIE: 0, Consts.P1: 0, Consts.P2: 0}

        # reset:
        game.reset()

        # update self-play
        if is_self_play and i % nUpdate == nUpdate - 1:
            player2 = copy.deepcopy(player1)
            player2.stop_learning()
    return log

# =================================================================================================================== #
def plot_log(log):
    plt.figure()
    plt.plot(log['episode'], log[Consts.TIE], 'g-', label='Tie')
    plt.plot(log['episode'], log[Consts.P1], 'b-', label='Player 1 wins')
    plt.plot(log['episode'], log[Consts.P2], 'r-', label='Player 2 wins')
    plt.title('Game Results')
    plt.ylabel('%')
    plt.xlabel('Episode')
    plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
    plt.show()

# =================================================================================================================== #
def evaluate(state, player1, player2, nGames=10000):
    winCount = {Consts.TIE: 0, Consts.P1: 0, Consts.P2: 0}
    game = Game(state, player1, player2)
    for i in tqdm(range(nGames)):
        if i - 1 < nGames/2 <= i:  # switch positions for half of the games
            game = Game(state, player2, player1)
        r1, r2, winner = game.play()
        if i >= nGames/2:  # players switched for this half of the games
            winner = switch_winner(winner)
        winCount[winner] += 1
        game.reset()
    return winCount[Consts.P1]/nGames, winCount[Consts.P2]/nGames, winCount[Consts.TIE]/nGames

def switch_winner(winner):
    if winner == Consts.P1:
        return Consts.P2
    elif winner == Consts.P2:
        return Consts.P1
    else:
        return Consts.TIE

# =================================================================================================================== #
if __name__ == '__main__':
    print("Auxiliary")

