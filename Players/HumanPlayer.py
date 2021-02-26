from Players.Player import Player

class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def get_action(self, state):
        command = state.get_action_command()
        command = self.name + ": " + command
        actionStr = input(command).split()
        action = tuple([float(a) for a in actionStr])
        return action

# =================================================================================================================== #

if __name__ == '__main__':
    print("Human player")

