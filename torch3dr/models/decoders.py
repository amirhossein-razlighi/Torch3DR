import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError(
            "BaseDecoder is an abstract class. Please implement the forward method in a subclass."
        )


class SimpleMLPVoxelDecoder(BaseDecoder):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(SimpleMLPVoxelDecoder, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))  # Give probabilities for each voxel (0 or 1)
        return x
