from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def get_action(self, state):
        pass

    def learn(self, reward, player):
        pass

    def stop_learning(self):
        pass

    def reset(self):
        pass

# =================================================================================================================== #

if __name__ == '__main__':
    print("Player")

