import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn
import math
from utils import MLP, load_buffer, to_np, to_torch
import gym
import d4rl
import mujoco_py
from shutil import copyfile

########################
######## model #########
########################

class EBMPriorTrainer:

    def __init__(self, prior, init_dist, lr=1e-2, magnitude_coeff=1):
        self.prior = prior
        self.init_dist = init_dist
        self.magnitude_coeff = magnitude_coeff
        self.optimizer = torch.optim.Adam(self.prior.parameters(), lr=lr)

    def train(self, data, batch_size=256, n_langevin_steps=10):
        positives = self.get_batch(data, batch_size)

        init = self.init_dist.sample((batch_size,))
        negatives = langevin_dynamics(self.prior, init, n_langevin_steps)
        negatives = negatives.detach()

        logp_positives = self.prior(positives)
        logp_negatives = self.prior(negatives)

        magnitude_regularization = logp_positives.pow(2).mean() + logp_negatives.pow(2).mean()

        loss = -logp_positives.mean() + logp_negatives.mean()
        loss += self.magnitude_coeff * magnitude_regularization

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), logp_positives, logp_negatives, magnitude_regularization.item()

    def get_batch(self, data, batch_size):
        inds = np.random.randint(0, len(data), size=batch_size)
        return data[inds]

def langevin_dynamics(model, init, n_steps, stepsize=0.01):
    x = init
    for i in range(n_steps):
        x = x.detach()
        x.requires_grad_()

        logp = model(x)
        grad_logp = torch.autograd.grad(logp.sum(), x)[0]

        noise = torch.randn(x.shape, device=x.device) * np.sqrt(2 * stepsize)
        x = x + stepsize * grad_logp + noise

    return x.detach()

########################
#### visualization #####
########################



def plot_samples(savepath, samples, plotting_index, xlim=None, ylim=None):
    print(f'Plotting {len(samples)} samples')
    samples = to_np(samples)
    plt.clf()
    plt.scatter(samples[:,plotting_index[0]], samples[:,plotting_index[1]], s=0.1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(plotting_index[0])
    plt.ylabel(plotting_index[1])
    plt.savefig(savepath)

def plot_hist(savepath, samples, plotting_index, xlim=None, ylim=None):
    print(f'Plotting histogram from {len(samples)} samples')
    samples = to_np(samples)
    plt.clf()
    plt.hist2d(samples[:,plotting_index[0]], samples[:,plotting_index[1]], range=[xlim, ylim], bins=100, cmap=plt.cm.jet)
    plt.xlabel(plotting_index[0])
    plt.ylabel(plotting_index[1])
    plt.savefig(savepath)


def plot_hopper(savepath, hopper_env, samples):
    samples = to_np(samples)
    np.random.shuffle(samples)
    hopper_env.reset()
    next_states = []
    for i in range(8):
        hopper_env.set_state(np.concatenate([[0], samples[i,:5]]), samples[i,5:11])
        next_state, _, _, _ = hopper_env.step(samples[i,11:])
        next_states.append(next_state)

    images = plot_hopper_states(hopper_env, np.array(next_states))
    fig, ax = plt.subplots(1, 8)
    for i in range(8):
      ax[i].imshow(images[i])
      ax[i].axis('off')
    plt.savefig(savepath)


def plot_hopper_states(hopper_env, states):
    viewer = mujoco_py.MjRenderContextOffscreen(hopper_env.sim)
    images = []
    for state in states:
        hopper_env.set_state(np.concatenate([[0],state[:5]]), state[5:])
        viewer.render(512, 512)
        data = viewer.read_pixels(512, 512, depth=True)
        # import IPython; IPython.embed()
        # data = data[::-1, :, :]
        # data = (data/255.).astype(np.float32)
        images.append(np.flip(data[1],0))
        # data = data.transpose()
        # plt.imshow(data)
        # plt.show()==
    return images

########################

## hyperparameters
magnitude_coeff = 0.01
n_langevin_steps = 100
n_training_steps = 500000

## just for visualization, probably don't need to change
n_visualization_samples = int(1e5)
n_visualization_steps = 1000

device = 'cuda:1'
logdir = 'data/hopper_mag_coeff_0pt01_stp_sz_0pt01'
os.makedirs(logdir, exist_ok=False)
copyfile("ebm.py", logdir+"/ebm.py")



#d4rl dataset
hopper = gym.make('hopper-medium-expert-v2')
dataset = hopper.get_dataset()
data_np = np.concatenate([dataset['observations'], dataset['actions']], axis = 1)
data = to_torch(data_np, device=device)

#get plotting indices
cov = np.cov(data_np, rowvar=False)
plotting_indices = np.array(list(zip([_ for _ in range((data_np).shape[1])], np.argmax(np.argsort(abs(cov)), axis=1))))


ebm = MLP(
    input_dim=data.shape[1],
    hidden_dims=[128, 128, 128],
    output_dim=1,
).to(device)

## initialization distribution for sampling
# init_dist = torch.distributions.Uniform(
#     data.min(dim=0)[0], data.max(dim=0)[0])
maxes = data.max(dim=0)[0]
mins = data.min(dim=0)[0]
ranges = maxes - mins
# init_dist = torch.distributions.Uniform(
   # mins - ranges / 2, maxes + ranges / 2)
init_dist = torch.distributions.MultivariateNormal(data.mean(dim=0), to_torch(cov, device))

trainer = EBMPriorTrainer(ebm, init_dist, magnitude_coeff=magnitude_coeff)

## visualize dataset
inds = np.random.choice(len(data), size=int(1e5))

for i in range(len(plotting_indices)):
    plot_samples(os.path.join(logdir, str(i)+'samples_data.png'), data[inds], plotting_indices[i])
    plot_hist(os.path.join(logdir, str(i)+'hist_data.png'), data[inds], plotting_indices[i])
plot_hopper(os.path.join(logdir, 'visual_data.png'), hopper, data[inds])

## visualize randomly-initialized model
init = init_dist.sample((n_visualization_samples,))

## train model
for j in range(n_training_steps):
    if j%1000 == 0:
        ## visualize trained model
        samples = langevin_dynamics(
            ebm, init, n_steps=n_visualization_steps)

        for i in range(len(plotting_indices)):
            plot_samples(os.path.join(logdir, str(i)+f'samples_{j}.png'),samples, plotting_indices[i])
            plot_hist(os.path.join(logdir, str(i)+f'hist_{j}.png'), samples, plotting_indices[i])
        plot_hopper(os.path.join(logdir, f'visual_{j}.png'), hopper, samples)
    loss, positives, negatives, magnitude = trainer.train(data, n_langevin_steps=n_langevin_steps)
    print(f'{j} | {loss:.6f} | {positives.mean():.6f} | {negatives.mean():.6f} | {magnitude:.6f}')


import IPython; IPython.embed()