import numpy as np
from matplotlib import pyplot as plt

class RPS:
    # 0=Rock, 1=Scissors, 2=Paper
    rewards = np.array([
        [0, 2, -1],
        [-1, 0, 2],
        [1, -2, 0]
    ])

class Player:
    def __init__(self, rewards, name):
        self.rewards = rewards
        self.regret_sum = np.zeros(len(rewards))
        self.strategy = np.ones(len(rewards))/len(rewards)
        self.strategy_sum = self.strategy.copy()
        self.name = name
    
    def update_action(self):
        positive_regret = self.regret_sum.clip(0)
        msum = np.sum(positive_regret)
        norm_regret = positive_regret/msum if msum > 0 else np.ones(len(self.rewards))/len(self.rewards)
        self.strategy = norm_regret
        self.strategy_sum += norm_regret

    def get_action(self):
        return np.random.choice(len(self.strategy), p=self.strategy)

    def add_regret(self, opp_strategy):
        lfunc = lambda x: opp_strategy.dot(x)
        reward = lfunc(self.rewards)
        counterfacts = reward.dot(self.strategy)
        self.regret_sum += (reward - counterfacts)
        print(self.regret_sum)

class Game:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.p1 = Player(RPS.rewards, 'joe')
        self.p2 = Player(RPS.rewards, 'bob')
        self.p1_gain = 0
        self.p2_gain = 0
    
    def play_game(self):
        history = np.zeros((self.max_iter, 2))
        for i in range(self.max_iter):
            print('=== iter {} ==='.format(i))
            a1 = self.p1.get_action()
            a2 = self.p2.get_action()
            self.p1_gain += RPS.rewards[a1, a2]
            self.p2_gain += RPS.rewards[a2, a1]
            print('p1 chose {}, p2 chose {}'.format(a1, a2))
            self.p1.add_regret(self.p2.strategy)
            self.p2.add_regret(self.p1.strategy)
            self.p1.update_action()
            self.p2.update_action()
            history[i, 0] = self.p1.strategy_sum[1]/self.p1.strategy_sum.sum()
            history[i, 1] = self.p2.strategy_sum[1]/self.p2.strategy_sum.sum()
        print('result: {}, {}'.format(self.p1_gain, self.p2_gain))
        plt.plot(history)
        plt.show()
    
g = Game(20000)
g.play_game()