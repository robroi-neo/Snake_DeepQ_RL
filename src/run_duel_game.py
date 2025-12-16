import numpy as np
import torch
from DuelDQN.game import SnakeGameAI
from DuelDQN.model import Dueling_QNet   # 👈 dueling model

# --- CONFIG ---
MODEL_PATH = "model/DuelDQN/checkpoint_2500.pth"


def load_model(path):
    """Load trained Dueling DQN model."""
    model = Dueling_QNet(16, 256, 3)  # input, hidden, output
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def get_action(model, state):
    """Convert model prediction into a one-hot action."""
    state_tensor = torch.tensor(state, dtype=torch.float)
    with torch.no_grad():
        q_values = model(state_tensor)

    move = torch.argmax(q_values).item()

    action = [0, 0, 0]
    action[move] = 1
    return action


def run():
    model = load_model()
    game = SnakeGameAI(fps=15)

    while True:
        state = game.get_state()
        action = get_action(model, state)

        _, game_over, score = game.play_step(action)

        if game_over:
            print("Final Score:", score)
            break


if __name__ == "__main__":
    run()
