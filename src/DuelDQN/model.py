import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Dueling_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 2, 8), 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 2, 8), output_size)
        )

    def forward(self, x):
        """Return Q-values for each action (batch, actions).

        Accepts a single sample tensor of shape (features,) or a batch (batch, features).
        Returns a tensor of shape (actions,) for single input, or (batch, actions) for batch input.
        """
        single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True

        x = self.feature(x)

        value = self.value(x)               # [batch, 1]
        advantage = self.advantage(x)       # [batch, actions]

        # Dueling aggregation: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))

        if single:
            return q_vals.squeeze(0)
        return q_vals

    def save(self, file_name='model.pth'):
        model_folder_path = './model/DuelDQN'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))



class QTrainer:
    def __init__(self, model, lr, gamma, device=None, target_update=100):
        self.lr = lr
        self.gamma = gamma
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        self.model = model.to(self.device)
        # create a target network for stable Q-learning
        self.target_model = type(model)(*self._model_init_args(model)).to(self.device)
        self.update_target(hard=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.target_update = target_update
        self.step_count = 0

    def _model_init_args(self, model):
        """Infer init args (input_size, hidden_size, output_size) from model layers."""
        input_size = model.feature[0].in_features
        hidden_size = model.feature[0].out_features
        output_size = model.advantage[-1].out_features
        return (input_size, hidden_size, output_size)

    def update_target(self, hard=False):
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            tau = 0.005
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    def train_step(self, state, action, reward, next_state, done):
        """Train on a batch (or single) of transitions.

        Accepts lists/tuples/numpy arrays or tensors.
        """
        # Convert to tensors on device
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done_mask = torch.tensor(done, dtype=torch.bool, device=self.device)

        action = torch.tensor(action, device=self.device)
        if action.dim() > 1:
            action_idx = action.argmax(dim=1).long()
        else:
            action_idx = action.long()

        # ensure batch dims
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            action_idx = action_idx.unsqueeze(0)
            done_mask = done_mask.unsqueeze(0)

        # predicted Q-values for current states
        pred_q = self.model(state)                # [batch, actions]
        q_pred = pred_q.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # [batch]

        # compute target Q-values using the target network
        with torch.no_grad():
            next_q = self.target_model(next_state)  # [batch, actions]
            max_next_q, _ = next_q.max(dim=1)       # [batch]
            q_target = reward + (~done_mask).float() * self.gamma * max_next_q

        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # periodic target update
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target(hard=True)

        return loss.item()