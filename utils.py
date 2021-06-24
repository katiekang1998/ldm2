import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.mujoco import HopperEnv
import torch.nn as nn
import torch 
from matplotlib import cm
import pickle


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.Tanh, output_activation=nn.Identity):
        super().__init__()

        layers = []
        current = input_dim
        for dim in hidden_dims:
            linear = nn.Linear(current, dim)
            layers.append(linear)
            layers.append(activation())
            current = dim

        layers.append(nn.Linear(current, output_dim))
        layers.append(output_activation())

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)




def load_buffer(loadpaths):
    data = []
    for loadpath in loadpaths:
        with open(loadpath, 'rb') as f:
            replay_buffer = pickle.load(f)
        if len(data)==0:
            data = np.concatenate([
                replay_buffer['observations'],
                replay_buffer['actions'],
                replay_buffer['next_observations'],
            ], axis=-1)
        else:
            new_stuff = np.concatenate([
                replay_buffer['observations'],
                replay_buffer['actions'],
                replay_buffer['next_observations'],
            ], axis=-1)
            data = np.concatenate([data, new_stuff])
    return data


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x

def to_torch(x, device = None):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, dtype=torch.float, device=device)