# Solving board games using Reinforcement Learning

In this personal project I aimed to implement a computer player for board games, using both heuristic and Reinforcement Learning (RL) based methods.

My main focus was on "Tic Tac Toe", but I also took my chances at "Connect 4".

## Usage

---

**Prerequisites:**

|   Library    |   Version   |
| :----------: | :---------: |
|   `Python`   |    `3.8`    |
|   `numpy`    | >= `1.18.5` |
|  `pytorch`   | >= `1.5.0`  |
| `matplotlib` | >= `3.2.1`  |
|    `tqdm`    | >= `4.50.2` |

**Start working:**

Download the project:

```
$ git clone https://github.com/roy-hachnochi/Reinforcement-Learning
```

The projects main results are concluded in 3 Jupyter notebooks. They may be run using Jupyter Notebook or using Google Colab. For the second option, download the entire repository into your Google Drive, in a subfolder named *RL*. Then, mount the drive using the following code block (which also appears in the notebooks):

```python
from google.colab import drive
drive.mount("/content/drive")
%cd ./drive/My\ Drive/RL
```

There is also a `Main.py` script for examples of manual running of the code, but this script is less organized and not so user friendly.

## Repository Organization

---

|             File/Directory             |                           Content                            |
| :------------------------------------: | :----------------------------------------------------------: |
|               `Players/`               |         Contains implementation of all player types.         |
|               `Utility/`               | Contains implementation of `Game` class, `train` and `evaluate` functions, and other utility functions and project constants. |
|              `TicTacToe/`              | Contains implementation of `TicTacToeState` class, and saved parameters of both deep and tabular Q-learning Tic Tac Toe players. |
|              `Connect4/`               | Contains implementation of `Connect4State` class, and saved parameters of the deep Q-learning Connect4 player. |
| `TicTacToe - Tabular Q-learning.ipynb` | Notebook containing all results for various types of training of the **tabular** Q-learning player on Tic Tac Toe. |
|  `TicTacToe - Deep Q-learning.ipynb`   | Notebook containing all results for various types of training of the **deep** Q-learning player on Tic Tac Toe. |
|   `Connect4 - Deep Q-learning.ipynb`   | Notebook containing all results for training of the deep Q-learning player on Connect 4. |
|               `Main.py`                |       Example script to run project without notebooks.       |

Note that all player and game classes are implemented generically so that any board game may be added merely by implementing a game state class, with the appropriate methods (see for example `TicTacToe.py` and `Connect4.py`).

## Players/Agents

---

1. `HumanPlayer`: The only non-computer player. Accepts action from user. Used for user interface.
2. `RandomPlayer`: Completely random player. Randomly chooses an action between all legal actions. Used for training and evaluation of the RL based players.
3. `SemiRandomPlayer`: Almost random player. If it has the ability to win in the current move - takes it, otherwise behaves as a random player. Used for training and evaluation of the RL based players.
4. `MaxMinPlayer`: Player based on the minimax algorithm. Looks `k` moves ahead, assuming that the opponent behaves perfectly, and chooses the next move with the best outcome based on some heuristic.
5. `QTPlayer`: Tabular Q-learning player.
6. `DQPlayer`: Deep Q-learning player.



### Minimax Player:

I won't get into explaining the entire Minimax algorithm, as many better and smarter have done it before. The main concept is that we model our game as a graph - where each node defines a state in the game, and nodes are connected by legal actions. The Minimax algorithm recursively unravels this graph by playing all possible moves against itself, in each move saving the best possible outcome for the current player. Once reaching a pre-defined maximum depth <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;d" title="d" /></a>, we stop and calculate the heuristic function, which is our definition of what counts as a "good" or "bad" state, we may see it as a sort of reward for reaching the state.

#### Minimax Heuristics:

- Tic Tac Toe:

  We first define a line heuristic <a href="https://www.codecogs.com/eqnedit.php?latex=h(l,t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(l,t)" title="h(l,t)" /></a> for some line <a href="https://www.codecogs.com/eqnedit.php?latex=l=[x_1,&space;x_2,&space;x_3]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l=[x_1,&space;x_2,&space;x_3]" title="l=[x_1, x_2, x_3]" /></a> and turn <a href="https://www.codecogs.com/eqnedit.php?latex=t\in{-1,1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t\in{-1,1}" title="t\in{-1,1}" /></a>:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=h(l)=\begin{cases}&space;G^{\sum_{x\in&space;l}&space;I(x=t)},&&space;\text{if&space;}&space;\sum_{x\in&space;l}&space;I(x=-t)=0\\&space;-G^{\sum_{x\in&space;l}&space;I(x=-t)},&&space;\text{if&space;}&space;\sum_{x\in&space;l}&space;I(x=t)=0\\&space;0,&space;&&space;\text{otherwise}&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(l)=\begin{cases}&space;G^{\sum_{x\in&space;l}&space;I(x=t)},&&space;\text{if&space;}&space;\sum_{x\in&space;l}&space;I(x=-t)=0\\&space;-G^{\sum_{x\in&space;l}&space;I(x=-t)},&&space;\text{if&space;}&space;\sum_{x\in&space;l}&space;I(x=t)=0\\&space;0,&space;&&space;\text{otherwise}&space;\end{cases}" title="h(l)=\begin{cases} G^{\sum_{x\in l} I(x=t)},& \text{if } \sum_{x\in l} I(x=-t)=0\\ -G^{\sum_{x\in l} I(x=-t)},& \text{if } \sum_{x\in l} I(x=t)=0\\ 0, & \text{otherwise} \end{cases}" /></a>
  
  We use <a href="https://www.codecogs.com/eqnedit.php?latex=G=10" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G=10" title="G=10" /></a>. In words - if there is only one kind of tile in the current line (line is still open for winning) we take its value as an exponent of the amount of tiles, and make it negative if it belongs to the opponent; otherwise (row contains both kinds of tiles) - the line has no value to any player so we set it to <a href="https://www.codecogs.com/eqnedit.php?latex=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0" title="0" /></a>.
  
  Now, the heuristic is defined as:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=H(s)=\sum_{l\in&space;s}&space;h(l),&space;\\&space;s=\{l_{row1},l_{row2},l_{row3},l_{col1},l_{col2},l_{col3},l_{diag1},l_{diag2}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H(s)=\sum_{l\in&space;s}&space;h(l),&space;\\&space;s=\{l_{row1},l_{row2},l_{row3},l_{col1},l_{col2},l_{col3},l_{diag1},l_{diag2}\}" title="H(s)=\sum_{l\in s} h(l), \\ s=\{l_{row1},l_{row2},l_{row3},l_{col1},l_{col2},l_{col3},l_{diag1},l_{diag2}\}" /></a>

- Connect 4:

  We use the same line-heuristic <a href="https://www.codecogs.com/eqnedit.php?latex=h(l)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(l)" title="h(l)" /></a> and similarly define:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=H(s)=\sum_{l\in&space;s}&space;h(l)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H(s)=\sum_{l\in&space;s}&space;h(l)" title="H(s)=\sum_{l\in s} h(l)" /></a>
  
  For connect 4, we take <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> to contain all groups of 4 adjacent slots in all directions (up-down, left-right, diagonal, and anti-diagonal).



### Tabular Q-learning:

In the Q-learning setting, we model the board game as a Markov Decision Process (MDP). In this setting, the states of the Markov Chain are all possible states of the board game, the transitions are based on the chosen action, where we model the opponent as part of the environment, meaning that the transition takes into account the chosen action of the player, and a **hidden** action which the opponent takes, which may be either random or deterministic.

The Q-leaning algorithm is based on the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Q_{n&plus;1}(s,a)=Q_n(s,a)&plus;\alpha((r(s,a)&plus;\gamma&space;max_{a'\in&space;A(s')}&space;\{Q(s',a')\})-Q_n(s,a))=\\&space;(1-\alpha)Q_n(s,a)&plus;\alpha(r(s,a)&plus;\gamma&space;max_{a'\in&space;A(s')}&space;\{Q(s',a')\})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Q_{n&plus;1}(s,a)=Q_n(s,a)&plus;\alpha((r(s,a)&plus;\gamma&space;max_{a'\in&space;A(s')}&space;\{Q(s',a')\})-Q_n(s,a))=\\&space;(1-\alpha)Q_n(s,a)&plus;\alpha(r(s,a)&plus;\gamma&space;max_{a'\in&space;A(s')}&space;\{Q(s',a')\})" title="Q_{n+1}(s,a)=Q_n(s,a)+\alpha((r(s,a)+\gamma max_{a'\in A(s')} \{Q(s',a')\})-Q_n(s,a))=\\ (1-\alpha)Q_n(s,a)+\alpha(r(s,a)+\gamma max_{a'\in A(s')} \{Q(s',a')\})" /></a>

Where:

- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;s" title="s" /></a> - current state.

- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a" title="a" /></a> - chosen action.

- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;s'" title="s'" /></a> - next state after taking action <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a" title="a" /></a> from state <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;s" title="s" /></a>.

- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;A(s')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;A(s')" title="A(s')" /></a> - action space for state <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;s'" title="s'" /></a>.

- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Q(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Q(s,a)" title="Q(s,a)" /></a> - state-action value function. Answers the question "what is our expected total reward for taking action <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a" title="a" /></a> at state <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;s" title="s" /></a>".

- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;r(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;r(s,a)" title="r(s,a)" /></a> - immediate reward for taking action <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a" title="a" /></a> at state <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;s" title="s" /></a>. We define:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;r(s,a)=\begin{cases}&space;r_{win},&&space;\text{if&space;}&space;s\in&space;S_{win}\\&space;r_{lose},&&space;\text{if&space;}&space;s\in&space;S_{lose}\\&space;r_{tie},&&space;\text{if&space;}&space;s\in&space;S_{tie}\\&space;0,&&space;\text{if&space;}&space;s\in&space;S_{non-terminal}\\&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;r(s,a)=\begin{cases}&space;r_{win},&&space;\text{if&space;}&space;s\in&space;S_{win}\\&space;r_{lose},&&space;\text{if&space;}&space;s\in&space;S_{lose}\\&space;r_{tie},&&space;\text{if&space;}&space;s\in&space;S_{tie}\\&space;0,&&space;\text{if&space;}&space;s\in&space;S_{non-terminal}\\&space;\end{cases}" title="r(s,a)=\begin{cases} r_{win},& \text{if } s\in S_{win}\\ r_{lose},& \text{if } s\in S_{lose}\\ r_{tie},& \text{if } s\in S_{tie}\\ 0,& \text{if } s\in S_{non-terminal}\\ \end{cases}" /></a>
  
- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\gamma" title="\gamma" /></a> - discount factor, how much value we give rewards which are achieved further into the game.

- <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a> - learning rate.

The idea is that we play many games and collect their resulting <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(s,a,r,s')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(s,a,r,s')" title="(s,a,r,s')" /></a> tuples for each step, based on the current approximation <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Q_n(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Q_n(s,a)" title="Q_n(s,a)" /></a>. At the end of each game we back-propagate the collected results using the above formula to update <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Q(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Q(s,a)" title="Q(s,a)" /></a>. Intuitively, the formula takes a weighted average of the current approximation, and a prediction of the maximal possible expected reward based on the gained experience and knowledge.



### Deep Q-learning:

While Tabular Q-learning is theoretically a good idea, and has even been proved to converge, it is actually impractical for most uses. The main reason is the giant state-action space. For example, even for a fairly simple game such as Connect 4, we have <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;|A|=7" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;|A|=7" title="|A|=7" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;|S|>=1.6\cdot10^{13}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;|S|>=1.6\cdot10^{13}" title="|S|>=1.6\cdot10^{13}" /></a>. Thus, assuming for example 4 bytes per Q-value, we have: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;|Q|>=4\cdot10^{14}\text{&space;}B\approx364\text{&space;}TB" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;|Q|>=4\cdot10^{14}\text{&space;}B\approx364\text{&space;}TB" title="|Q|>=4\cdot10^{14}\text{ }B\approx364\text{ }TB" /></a>. Moreover, for the Q-table to truly converge, we must visit each of the above state-action pairs a sufficiently large amount of times, which makes it infeasible... Furthermore, a Q-table forces us to have discrete state and action spaces, which isn't always the case.

Deep Q-learning comes to solve these problems. Instead of saving a table <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Q(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Q(s,a)" title="Q(s,a)" /></a> for each <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(s,a)" title="(s,a)" /></a> pair, we look at this table as a function, and try to approximate it. More precisely, we have: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Q_\theta(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Q_\theta(s,a)" title="Q_\theta(s,a)" /></a> which is a function of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(s,a)" title="(s,a)" /></a>, and evaluated using parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta" title="\theta" /></a>. Our function approximator will be a deep neural network, with weights <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta" title="\theta" /></a>. Now, instead of updating the Q-values themselves, we update <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta" title="\theta" /></a> based on a gradient descent of the above equation:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_{n&plus;1}=\theta_n-\alpha\nabla_{\theta}L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_{n&plus;1}=\theta_n-\alpha\nabla_{\theta}L" title="\theta_{n+1}=\theta_n-\alpha\nabla_{\theta}L" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L=(Q_{\theta_n}(s,a)-(r(s,a)&plus;\gamma&space;max_{a'\in&space;A(s')}&space;\{Q_{\theta_n}(s',a')\}))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L=(Q_{\theta_n}(s,a)-(r(s,a)&plus;\gamma&space;max_{a'\in&space;A(s')}&space;\{Q_{\theta_n}(s',a')\}))^2" title="L=(Q_{\theta_n}(s,a)-(r(s,a)+\gamma max_{a'\in A(s')} \{Q_{\theta_n}(s',a')\}))^2" /></a>

This is just the MSE of the predicted Q-value and the estimated one.

The two main advantages of this approach:

- The Q-function becomes scalable. In fact, it doesn't matter now how big the state-action space is, we will always stay with a constant amount of parameters to calculate (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta" title="\theta" /></a>).
- We also hope to gain some generalization between states. Instead of each state-action pair being solely dependent on its own experience, this setting allows us to generalize knowledge gained from previous experience into never-before-seen states. This leans on the assumption that different states actually have some general similarity which may be exploited.

That being said, it is well known that Deep Q-networks are difficult to train. There are many improvements done in this field, such as target Q-networks, experience replay, and more, some of which I have implemented.

## Results

---

Detailed results are concluded in the three notebooks, including results for various training techniques. A summary is supplied here:

|                                                |                   Random                    |                 Semi-Random                  |               Minimax                |
| :--------------------------------------------: | :-----------------------------------------: | :------------------------------------------: | :----------------------------------: |
| **Tic Tac Toe -** <br />**Tabular Q-learning** | Win: 93.17%<br />Lose: 0.2%<br />Tie: 6.63% |                      -                       | Win: 0%<br />Lose: 0%<br />Tie: 100% |
|   **Tic Tac Toe -**<br />**Deep Q-learning**   |                      -                      | Win: 94.59%<br />Lose: 0.29%<br />Tie: 5.12% | Win: 0%<br />Lose: 0%<br />Tie: 100% |
|    **Connect 4 -**<br />**Deep Q-learning**    |                      -                      |  Win: 63.62%<br />Lose: 36.38%<br />Tie: 0%  | Win: 0%<br />Lose: 100%<br />Tie: 0% |

As we can see, the results are close to perfect for Tic Tac Toe, but not nearly as good for Connect 4. There is much probably work to do here to make this work, either in the training, the architecture, the hyperparameters, or the algorithm itself. But all of this for another time...