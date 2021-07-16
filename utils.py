import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.mujoco import HopperEnv
import torch.nn as nn
import torch 
from matplotlib import cm
import pickle
from rendering import Renderer
import environments

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.Tanh, output_activation=nn.Identity):
#         super().__init__()

#         layers = []
#         current = input_dim
#         for dim in hidden_dims:
#             linear = nn.Linear(current, dim)
#             layers.append(linear)
#             layers.append(activation())
#             current = dim

#         layers.append(nn.Linear(current, output_dim))
#         layers.append(output_activation())

#         self._layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self._layers(x)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


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

def to_torch(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, dtype=torch.float).cuda()

def plot_samples(savepath, samples):
    renderer = Renderer('HopperFullObs-v2')
    render_kwargs = {
        'trackbodyid': 2,
        'distance': 10,
        'lookat': [10, 2, 0.5],
        'elevation': 0
    }
    renderer.composite(savepath, to_np(samples), dim=(1024, 256), partial = True, qvel=True, render_kwargs=render_kwargs)

def process_pendulum_obs(obs):
    theta_cos = obs[:, 0]
    theta_sin = obs[:, 1]
    theta_dot = obs[:, 2]
    theta = np.arctan2(theta_sin, theta_cos)
    return np.stack([theta, theta_dot], axis=1)

def plot_pendulum_densities(density_model):
    theta_coordinates = np.linspace(-np.pi, np.pi, 100)
    theta_dot_coordinates = np.linspace(-8, 8, 110)
    actions_coordinates = np.linspace(-1, 1, 12)

    grid = []
    for i in theta_coordinates:
        for j in theta_dot_coordinates:
            for k in actions_coordinates:
                grid.append([i, j, k])

    len_dataset = len(grid)
    data = to_torch(grid)
    rew = np.zeros(len_dataset)
    for i in range(len_dataset//256):
      rew[i*256:(i+1)*256] = density_model(data[i*256:(i+1)*256], return_np=True)
    rew[(len_dataset//256)*256:]=density_model(data[(len_dataset//256)*256:], return_np=True)


    ebm_output = rew.reshape(100, 110, 12)
    state_ebm_outputs = np.log(np.sum(np.e**ebm_output, axis=2))
    plt.close()
    # plt.contourf(theta_coordinates, theta_dot_coordinates, np.transpose(state_ebm_outputs))
    plt.imshow(state_ebm_outputs, cmap=cm.jet, extent=[theta_coordinates.min(), theta_coordinates.max(), theta_dot_coordinates.min(), theta_dot_coordinates.max()], aspect='auto')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

def plot_pendulum_q(critic):
    theta_coordinates = np.linspace(-np.pi, np.pi, 100)
    theta_dot_coordinates = np.linspace(-8, 8, 110)
    actions_coordinates = np.linspace(-1, 1, 12)

    states = []
    actions = []
    for i in theta_coordinates:
        for j in theta_dot_coordinates:
            for k in actions_coordinates:
                states.append([i, j])
                actions.append([k])

    len_dataset = len(actions)
    states = to_torch(states)
    actions = to_torch(actions)
    rew = np.zeros(len_dataset)
    for i in range(len_dataset//256):
      rew[i*256:(i+1)*256] = to_np(critic(states[i*256:(i+1)*256], actions[i*256:(i+1)*256])[0]).squeeze()
    rew[(len_dataset//256)*256:]=to_np(critic(states[(len_dataset//256)*256:], actions[(len_dataset//256)*256:])[0]).squeeze()


    ebm_output = rew.reshape(100, 110, 12)
    state_ebm_outputs = np.log(np.sum(np.e**ebm_output, axis=2))
    plt.close()
    # plt.contourf(theta_coordinates, theta_dot_coordinates, np.transpose(state_ebm_outputs))
    plt.imshow(state_ebm_outputs, cmap=cm.jet, extent=[theta_coordinates.min(), theta_coordinates.max(), theta_dot_coordinates.min(), theta_dot_coordinates.max()], aspect='auto')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()