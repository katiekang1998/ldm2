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

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, ebm):
        super().__init__()

        self.Q1 = copy.deepcopy(ebm)
        self.Q2 = copy.deepcopy(ebm)

        self.outputs = dict()

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

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

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

class SACAgent():
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range,
                 actor_lr, actor_betas, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, ebm ):
        super().__init__()

        self.action_range = action_range
        self.critic_tau = critic_tau
        self.batch_size = batch_size
        self.critic_target_update_frequency = critic_target_update_frequency
        self.obs_dim = obs_dim

        self.critic = DoubleQCritic(ebm).cuda()
        self.critic_target = DoubleQCritic(ebm).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim, action_dim, 1024, 2, [-5, 2]).cuda()
        self.ebm = ebm

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.train()
        self.critic_target.train()


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).cuda()
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        # assert action.ndim == 2 and action.shape[0] == 1
        return action[0].cpu().detach().numpy()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        if self.ebm:
            ebm_values = self.ebm(torch.cat([next_obs, next_action], dim=1))
            target_V = torch.min(ebm_values, torch.min(target_Q1, target_Q2))
        else:
            target_V = torch.min(target_Q1,
                                 target_Q2) 
        target_Q = torch.min(torch.stack([reward, target_V]), dim=0)[0]
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        print(critic_loss)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        if self.ebm:
            ebm_values = self.ebm(torch.cat([obs, action], dim=1))
            actor_Q = torch.min(ebm_values, torch.min(actor_Q1, actor_Q2))
        else:
            actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = ( - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def plot_values(self, threshold = -6):
        places_evaluated_np = []
        for i in range(-310, 320):
          for j in range(-80, 90):
            places_evaluated_np.append([0.01*i, 0.1*j])
        places_evaluated = torch.from_numpy(np.array(places_evaluated_np)).type(torch.FloatTensor).cuda()
        # places_evaluated_energies = self.critic.Q1(places_evaluated)
        dist = self.actor(places_evaluated)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(places_evaluated, next_action)
        target_V = torch.min(target_Q1, target_Q2)
        places_evaluated_energies_np = target_Q1.detach().cpu().numpy().reshape((630, 170))

        masked_V = np.ma.masked_where(places_evaluated_energies_np >= threshold, places_evaluated_energies_np)
        cmap = matplotlib.cm.viridis
        cmap.set_bad(color='white')

        plt.imshow(np.transpose(-masked_V), aspect=32./6., cmap='jet')
        plt.colorbar()
        plt.show()

    def update(self, step, replay_buffer):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(
            self.batch_size)

        self.update_critic(obs, action, reward, next_obs, not_done)

        self.update_actor(obs)
        

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)


    def update_actor_only(self, step, replay_buffer):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(
            self.batch_size)

        # self.update_critic(obs, action, reward, next_obs, not_done)

        self.update_actor(obs)
        

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def load(self, actor_f, critic_f, critic_target_f):
        self.actor.load_state_dict(torch.load(actor_f))
        self.critic.load_state_dict(torch.load(critic_f))
        self.critic_target.load_state_dict(torch.load(critic_target_f))

#plotting
    def get_ldm_values(self, data):
        data_torch = torch.from_numpy(data[:, :self.obs_dim]).type(torch.FloatTensor).cuda()  
        action_dist = self.actor(data_torch)
        best_actions = action_dist.rsample()
        target_Q1s, target_Q2s = self.critic_target(data_torch, best_actions)   
        ldm_data_values = torch.min(target_Q1s, target_Q2s).cpu().detach().numpy().squeeze()
        return ldm_data_values


    def get_bootstrapping_error(self, obs, action, reward, next_obs):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2)
        target_Q = torch.min(torch.stack([reward, target_V]), dim=0)[0]
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        return critic_loss

