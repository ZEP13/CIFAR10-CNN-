import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5) #(12, 28, 28)
        self.pool = nn.MaxPool2d(2,2) #(12, 14, 14)
        self.conv2 = nn.Conv2d(12, 24, 5) #(24, 10, 10) -> Flatten (24, 5, 5)
        self.fc1 = nn.Linear(24*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(path='model/train_net.pth'):
    model = NeuralNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
