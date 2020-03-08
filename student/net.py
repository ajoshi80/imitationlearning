import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(6 * 6 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.permute(0,3,1,2).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)
    
    def get_action(self, inp):
        inp = torch.from_numpy(inp).float()
        inp = torch.reshape(inp, (1, inp.shape[0], inp.shape[1], inp.shape[2]))
        out = self.forward(inp).squeeze()
        best_action = int(torch.argmax(out))
        return best_action

