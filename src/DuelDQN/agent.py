import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Dueling_QNet, QTrainer

from helper import save_plot

import os
os.makedirs("model/DuelDQN", exist_ok=True)


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Dueling_QNet(16, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        # store action as index (0/1/2) for compactness; accept one-hot or index
        if isinstance(action, (list, tuple)):
            action_idx = int(np.argmax(action))
        else:
            action_idx = int(action)
        self.memory.append((state, action_idx, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # actions are stored as indices
        self.trainer.train_step(states, list(actions), list(rewards), next_states, list(dones))
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        # accept one-hot or index
        if isinstance(action, (list, tuple)):
            action_idx = int(np.argmax(action))
        else:
            action_idx = int(action)
        self.trainer.train_step(state, action_idx, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.n_games >= 1000:
            epsilon = 0
        else:
            epsilon = max(5, 80 - self.n_games)  # exploration factor before 1000 games

        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            record = max(record, score)
            # if score - record >= 30:  # breakthrough threshold
            #    agent.model.save(f"breakthrough_{score}.pth")
            
            if agent.n_games % 100 == 0:
                agent.model.save(f"checkpoint_{agent.n_games}.pth")
                print(f"Checkpoint saved at game {agent.n_games}")

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'frames', game.frame_iteration)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            save_plot(plot_scores, plot_mean_scores, agent.n_games)


if __name__ == '__main__':
    train()