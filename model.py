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
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
            # Convert inputs to tensors if they are not already
        state = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float)
        next_state = next_state if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, dtype=torch.float)
        action = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        reward = reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float)
        done = done if isinstance(done, torch.Tensor) else torch.tensor(done, dtype=torch.bool)

        # Ensure inputs are in batch form
        state = state.unsqueeze(0) if state.dim() == 1 else state
        next_state = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
        action = action.unsqueeze(0) if action.dim() == 1 else action
        reward = reward.unsqueeze(0) if reward.dim() == 1 else reward
        done = done.unsqueeze(0) if done.dim() == 0 else done  # Ensure 'done' is a 1D tensor for batch processing

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(done.size(0)):  # Iterate over batch
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


