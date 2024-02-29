import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

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

        # Ensure that state and next_state are tensors with the required shape
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0) if not isinstance(state, torch.Tensor) else state
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0) if not isinstance(next_state, torch.Tensor) else next_state

        # Ensure action and reward are wrapped in tensors
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)

        # Predict Q values for the current state
        pred = self.model(state)

        # Clone the predictions to use as the target
        target = pred.clone()

        # Compute the new Q value
        Q_new = reward.item()
        if not done:
            Q_new += self.gamma * torch.max(self.model(next_state).detach())

        # Update the target for the action taken
        target[0][action] = Q_new

        # Zero the gradients
        self.optimizer.zero_grad()

        # Calculate the loss
        loss = self.criterion(target, pred)

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        self.optimizer.step()


