import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import math
import matplotlib
import copy
from utils import *


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist


class BC():
    def __init__(self, obs_dim, action_dim, action_range,
                 actor_lr, actor_betas, batch_size):
        super().__init__()

        self.action_range = action_range
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = DiagGaussianActor(obs_dim, action_dim, 1024, 2, [-5, 2]).cuda()

                # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).cuda()
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        # assert action.ndim == 2 and action.shape[0] == 1
        return action[0].cpu().detach().numpy()
                                                                                                                                                                   
    def update_actor(self, obs, action):
        dist = self.actor(obs)
        action = action.clamp(*np.array(self.action_range)*0.999)
        action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        loss = (-action_log_prob).mean()

        if torch.isnan(loss):
            import IPython; IPython.embed()
        # optimize the actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update(self, step, replay_buffer):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(
            self.batch_size)
        self.update_actor(obs, action)
        
    def load(self, actor_f):
        self.actor.load_state_dict(torch.load(actor_f))

