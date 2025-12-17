import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model/DQN'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, device=None, target_update=100):
        self.lr = lr
        self.gamma = gamma
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        self.model = model.to(self.device)
        # create target network
        self.target_model = type(model)(*self._model_init_args(model)).to(self.device)
        self.update_target(hard=True)

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # use Huber loss (more robust)
        self.criterion = nn.SmoothL1Loss()

        self.target_update = target_update
        self.step_count = 0

    def _model_init_args(self, model):
        input_size = model.linear1.in_features
        hidden_size = model.linear1.out_features
        output_size = model.linear2.out_features
        return (input_size, hidden_size, output_size)

    def update_target(self, hard=False):
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            tau = 0.005
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    def train_step(self, state, action, reward, next_state, done):
        # support lists/tuples/numpy arrays
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

        # ensure batch dims
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done_mask = done_mask.unsqueeze(0)

        # predicted Q-values for current states
        pred_q = self.model(state)                # [batch, actions]

        # parse action input (support: scalar index, one-hot vector, batch of indices, batch of one-hot vectors)
        action = action.to(self.device)
        if action.dim() == 0:
            action_idx = action.long().unsqueeze(0)
        elif action.dim() == 1:
            # one-hot single vector (length == n_actions)
            if action.numel() == pred_q.size(1):
                action_idx = action.argmax().unsqueeze(0).long()
            # batch of indices (length == batch)
            elif action.numel() == state.size(0):
                action_idx = action.long()
            else:
                action_idx = action.long()
        else:
            # assume one-hot batch: (batch, n_actions)
            action_idx = action.argmax(dim=1).long()

        q_pred = pred_q.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # [batch]

        # Double DQN target: select best action by online network, evaluate with target network
        with torch.no_grad():
            next_actions = self.model(next_state).argmax(dim=1, keepdim=True)  # [batch,1]
            next_q_target = self.target_model(next_state).gather(1, next_actions).squeeze(1)
            q_target = reward + (~done_mask).float() * self.gamma * next_q_target

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