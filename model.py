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

        # Convert single instances to tensors and add a batch dimension
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0) if not isinstance(state, torch.Tensor) else state
        action = torch.tensor([action], dtype=torch.long) if not isinstance(action, torch.Tensor) else action
        reward = torch.tensor([reward], dtype=torch.float) if not isinstance(reward, torch.Tensor) else reward
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0) if not isinstance(next_state, torch.Tensor) else next_state
        done = torch.tensor([done], dtype=torch.bool) if not isinstance(done, torch.Tensor) else done

        # Predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(done.size(0)):
            Q_new = reward[idx].item()
            if not done[idx]:
                Q_new += self.gamma * torch.max(self.model(next_state[idx]).detach())
            action_idx = action[idx].item()  # Ensure action_idx is a scalar
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


