from Players.HumanPlayer import HumanPlayer
from Utility.Consts import Consts
import time
import matplotlib.pyplot as plt

class Game:
    def __init__(self, state, p1, p2):
        self.state = state
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False

    def play_turn(self):
        player = self.p1 if self.state.curPlayer == Consts.P1 else self.p2
        action = player.get_action(self.state)
        succ, isEnd = self.state.play(action)
        while not succ:
            print("Warning: Player turn failed. Retrying...")
            action = player.get_action(self.state)
            succ, isEnd = self.state.play(action)
        self.isEnd = isEnd

    def play(self):
        isHuman = isinstance(self.p1, HumanPlayer) or isinstance(self.p2, HumanPlayer)
        if isHuman:
            print("############################################################################################")
            print(self.state.name)
            print("############################################################################################")
        while not self.isEnd:
            if isHuman:
                self.state.show_state()
            self.play_turn()
        _, winner = self.state.check_win()
        if isHuman:
            self.state.show_state()
            if winner == Consts.P1:
                print("{} wins!".format(self.p1.name))
            if winner == Consts.P2:
                print("{} wins!".format(self.p2.name))
            if winner == Consts.TIE:
                print("Tie!")
        r1, r2 = self.state.get_rewards()
        self.p1.learn(r1, Consts.P1)
        self.p2.learn(r2, Consts.P2)
        return r1, r2, winner

    def reset(self):
        self.state.reset()
        self.p1.reset()
        self.p2.reset()
        self.isEnd = False

# =================================================================================================================== #
def train(state, player1, player2, nGames=100000):
    nLog = 1000
    winCount = {Consts.TIE: 0, Consts.P1: 0, Consts.P2: 0}
    log = {Consts.TIE: [], Consts.P1: [], Consts.P2: [], 'episode': []}
    startTime = time.time()
    for i in range(nGames):
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
        if i % 1000 == 999:
            endTime = time.time()
            print('Episode {}: player 1 reward = {} | player 2 reward = {}. Elapsed time: {:.0f} secs'.
                  format(i, r1, r2, endTime - startTime))
            startTime = endTime
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
def evaluate(state, player1, player2, nGames=100000):
    winCount = {Consts.TIE: 0, Consts.P1: 0, Consts.P2: 0}
    startTime = time.time()
    game = Game(state, player1, player2)
    for i in range(nGames):
        r1, r2, winner = game.play()
        winCount[winner] += 1
        game.reset()
        if i % 1000 == 999:
            endTime = time.time()
            print('Episode {} done. Elapsed time: {:.0f} secs'.format(i, endTime - startTime))
            startTime = endTime
    return winCount[Consts.P1]/nGames, winCount[Consts.P2]/nGames, winCount[Consts.TIE]/nGames

# =================================================================================================================== #
if __name__ == '__main__':
    print("Game")

