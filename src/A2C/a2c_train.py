# a2c_train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from a2c_model import ActorCritic

from game import SnakeGameAI
import numpy as np

UPDATE_EVERY = 20


GAMMA = 0.99
LR = 0.0005

def select_action(model, state):
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    
    # Forward pass through actor-critic
    logits, value = model(state)
    
    # Convert logits to probability distribution
    probs = F.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)

    # Sample an action
    action = dist.sample()

    # Compute log probability and entropy for training
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    # MUST return entropy so the training loop can use it
    return action.item(), log_prob, value, entropy


def compute_returns(rewards, values, gamma=GAMMA):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float)
    values = torch.cat(values).squeeze()
    advantages = returns - values.detach()
    return returns, advantages

def train_a2c():
    game = SnakeGameAI(w=10, h=10, fps=20)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode = 0

    while True:
        state = game.get_state()

        log_probs = []
        values = []
        rewards = []
        entropies = []

        done = False

        while not done:
            action_idx, log_prob, value, entropy = select_action(model, state)

            # convert index to move-array
            action = [0, 0, 0]
            action[action_idx] = 1

            reward, done, score = game.play_step(action)
            next_state = game.get_state()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy) 

            state = next_state

        # Episode finished
        game.reset()
        episode += 1

        # Compute returns & advantages
        returns, advantages = compute_returns(rewards, values)

        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()

        # Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss * 0.5

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode} | Score: {score} | Loss: {loss.item():.4f}")

        # Optional save checkpoint
        if episode % 1000 == 0:
            torch.save(model.state_dict(), f"./model/A2C/a2c_{episode}.pth")
            print("Checkpoint saved.")

if __name__ == "__main__":
    train_a2c()
