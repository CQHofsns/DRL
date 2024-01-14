import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        self.relu= nn.ReLU()
        
        self.layer1= nn.Linear(4, 128)
        self.layer2= nn.Linear(128, 128)
        self.layer3= nn.Linear(128, n_actions)

    def forward(self, x):
        x= self.relu(self.layer1(x))
        x= self.relu(self.layer2(x))
        x= self.layer3(x)

        return x

def load_model(n_actions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net= DQN(n_actions= n_actions).to(device)
    policy_net.load_state_dict(torch.load('save_models/policy_model_state_dict_CartPole_v1_600episodes_v1.pt'))
    
    return policy_net