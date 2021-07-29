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

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

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

class LDMSAC():
    def __init__(self, obs_dim, action_dim, action_range,
                 actor_lr, actor_betas, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, sac_coefficient, density_model=None, cql_regularizer_coefficient=0, state_dist=None, action_dist=None, discount = 0.99, init_temperature= 0.1, alpha_lr= 1e-4, alpha_betas= [0.9, 0.999]):
        super().__init__()

        self.action_range = action_range
        self.critic_tau = critic_tau
        self.batch_size = batch_size
        self.critic_target_update_frequency = critic_target_update_frequency
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cql_regularizer_coefficient = cql_regularizer_coefficient

        self.sac_coefficient = sac_coefficient

        self.density_model = density_model

        self.critic_ldm = DoubleQCritic(obs_dim, action_dim, 1024, 2).cuda()
        self.critic_target_ldm = DoubleQCritic(obs_dim, action_dim, 1024, 2).cuda()

        self.critic_sac = DoubleQCritic(obs_dim, action_dim, 1024, 2).cuda()
        self.critic_target_sac = DoubleQCritic(obs_dim, action_dim, 1024, 2).cuda()

        self.critic_optimizer_ldm = torch.optim.Adam(self.critic_ldm.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)
        self.critic_optimizer_sac = torch.optim.Adam(self.critic_sac.parameters(),
                                         lr=critic_lr,
                                         betas=critic_betas)

        # self.pretrain_critic(state_dist=state_dist, action_dist=action_dist)
        # import IPython; IPython.embed()

        self.critic_target_ldm.load_state_dict(self.critic_ldm.state_dict())
        self.critic_target_sac.load_state_dict(self.critic_sac.state_dict())

        self.actor = DiagGaussianActor(obs_dim, action_dim, 1024, 2, [-5, 2]).cuda()

                # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)





        self.discount = discount

        self.log_alpha = torch.tensor(np.log(init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target_ldm.train()
        self.critic_target_sac.train()


    # def pretrain_critic(self, state_dist = pyd.uniform.Uniform(to_torch([-np.pi, -8]), to_torch([np.pi, 8])), action_dist = pyd.uniform.Uniform(to_torch([-1]), to_torch([1]))):
    #     for step in range(15000):
    #         states_uniform = state_dist.sample(sample_shape = [128])
    #         actions_uniform = action_dist.sample(sample_shape = [128])
    #         samples = self.density_model.model.sample(128)[-1]
    #         states_samples = samples[:, :len(states_uniform[0])]
    #         actions_samples = samples[:, len(states_uniform[0]):]
    #         states = torch.cat([states_uniform, states_samples], dim=0)
    #         actions = torch.cat([actions_uniform, actions_samples], dim=0)
    #         x = torch.cat([states, actions], dim=1)
    #         densities = torch.clamp(self.density_model(x), -200, 200)
    #         q1, q2 = self.critic(states, actions)
    #         loss = torch.mean((q1-densities)**2+(q2-densities)**2)
    #         print(step)
    #         print(loss)
    #         self.critic_optimizer.zero_grad()
    #         loss.backward()
    #         self.critic_optimizer.step()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic_ldm.train(training)
        self.critic_sac.train(training)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).cuda()
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        # assert action.ndim == 2 and action.shape[0] == 1
        return action[0].cpu().detach().numpy()

    def update_critic_ldm(self, obs, action, reward, next_obs, not_done):
        reward = reward.unsqueeze(-1)
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        target_Q1, target_Q2 = self.critic_target_ldm(next_obs, next_action)
        if self.density_model:
            densities = torch.clamp(self.density_model(torch.cat([next_obs, next_action], dim=1)), -200, 200)
            target_V = torch.min(densities, torch.min(target_Q1, target_Q2))
        else:
            target_V = torch.min(target_Q1,
                                 target_Q2) 
        target_Q = torch.min(torch.stack([reward, target_V]), dim=0)[0]
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic_ldm(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        print("critic ldm loss")
        print(critic_loss)
        
        if torch.isnan(critic_loss):
            import IPython; IPython.embed()
        # Optimize the critic
        self.critic_optimizer_ldm.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_ldm.step()


    def update_critic_sac(self, obs, action, reward, next_obs, not_done):

        reward = reward.unsqueeze(-1)
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target_sac(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic_sac(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        print("critic reward loss")
        print(critic_loss)

        # Optimize the critic
        self.critic_optimizer_sac.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_sac.step()
                                                                                                                                                                                                               
    def update_actor(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        actor_Q1_ldm, actor_Q2_ldm = self.critic_ldm(obs, action)
        if self.density_model:
            densities = torch.clamp(self.density_model(torch.cat([obs, action], dim=1)), -200, 200)
            actor_Q_ldm = torch.min(densities, torch.min(actor_Q1_ldm, actor_Q2_ldm))
        else:
            actor_Q_ldm = torch.min(actor_Q1_ldm, actor_Q2_ldm)

        actor_Q1_sac, actor_Q2_sac = self.critic_sac(obs, action)
        actor_Q_sac = torch.min(actor_Q1_sac, actor_Q2_sac)

        print("actor LDM loss")
        print((-actor_Q_ldm).mean())

        print("actor reward loss")
        print(self.sac_coefficient * (-actor_Q_sac).mean())

        print("actor params")
        print(torch.isnan(dist.loc).sum())
        print(torch.isnan(dist.scale).sum())

        actor_loss = (-actor_Q_ldm).mean() + self.sac_coefficient * (-actor_Q_sac).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_actor_with_density_model(self, obs):
        dist = self.actor(obs)
        print(dist.loc)
        print(dist.scale)
        print(" ")
        action = dist.rsample()
        densities = torch.clamp(self.density_model(torch.cat([obs, action], dim=1)), -200, 200)
        actor_loss = (-densities).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def update(self, step, replay_buffer):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(
            self.batch_size)

        self.update_critic_ldm(obs, action, reward[:, 0], next_obs, not_done)

        self.update_critic_sac(obs, action, reward[:, 1], next_obs, not_done)

        self.update_actor(obs)
        

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic_ldm, self.critic_target_ldm,
                                     self.critic_tau)
            soft_update_params(self.critic_sac, self.critic_target_sac,
                                     self.critic_tau)


    def update_actor_only(self, step, replay_buffer):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(
            self.batch_size)

        # self.update_critic(obs, action, reward, next_obs, not_done)

        # self.update_actor(obs)
        self.update_actor_with_density_model(obs)


    def load(self, actor_f, critic_f, critic_target_f):
        self.actor.load_state_dict(torch.load(actor_f))
        self.critic.load_state_dict(torch.load(critic_f))
        self.critic_target.load_state_dict(torch.load(critic_target_f))

    def get_ldm_values(self, data):
        data_torch = torch.from_numpy(data[:, :self.obs_dim]).type(torch.FloatTensor).cuda()  
        action_dist = self.actor(data_torch)
        best_actions = action_dist.rsample()
        target_Q1s, target_Q2s = self.critic_target(data_torch, best_actions)   
        ldm_data_values = torch.min(target_Q1s, target_Q2s).cpu().detach().numpy().squeeze()
        return ldm_data_values


    def get_td_error(self, obs, action, reward, next_obs):
        obs = to_torch(obs)
        action = to_torch(action)
        reward = to_torch(reward).unsqueeze(1)
        next_obs = to_torch(next_obs)
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

        return to_np((current_Q1- target_Q)**2 + (current_Q2- target_Q)**2).squeeze()

