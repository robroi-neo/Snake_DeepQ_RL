import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import save_plot

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 1e-4

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(16, 256, 3)
        # use larger target_update for stability
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, target_update=500)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
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

    plot_survival = []
    plot_mean_survival = []
    total_survival = 0

    survival_steps = 0

    agent = Agent()
    game = SnakeGameAI()
    while True:

        survival_steps += 1
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

            print('Game', agent.n_games, 'Score', score, 'Record:', record,)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            save_plot(
                plot_scores,
                plot_mean_scores,
                agent.n_games,
                ylabel="Score",
                title="Score vs Games",
                prefix="score"
            )


            plot_survival.append(survival_steps)
            total_survival += survival_steps
            mean_survival = total_survival / agent.n_games
            plot_mean_survival.append(mean_survival)

            survival_steps = 0
            save_plot(
                plot_survival,
                plot_mean_survival,
                agent.n_games,
                ylabel="Survival Steps",
                title="Survival vs Games",
                prefix="survival"
            )



if __name__ == '__main__':
    train()