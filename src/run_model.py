import numpy as np
import torch
from game import SnakeGameAI
from model import Linear_QNet

# --- CONFIG ---
MODEL_PATH = "model/model.pth"

def load_model():
    """Load trained model from file."""
    model = Linear_QNet(16, 256, 3)  # input, hidden, output
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def get_action(model, state):
    """Convert model prediction into a one-hot action."""
    state_tensor = torch.tensor(state, dtype=torch.float)
    prediction = model(state_tensor)

    move = torch.argmax(prediction).item()

    action = [0, 0, 0]
    action[move] = 1
    return action


def run():
    model = load_model()
    game = SnakeGameAI(fps=200)

    while True:
        state = game.get_state()
        action = get_action(model, state)

        _, game_over, score = game.play_step(action)

        if game_over:
            print("Final Score:", score)
            break


if __name__ == "__main__":
    run()