import torch.nn as nn
import numpy as np
from utils import *
import itertools


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter
from bc import BC



class BCDensityModel(nn.Module):
	def __init__(self, file, state_dim, action_dim):
		super(BCDensityModel, self).__init__()

		# construct the model
		self.model = BC(state_dim, action_dim, [-1, 1],
				 1e-4, [0.9, 0.999], 1024)
		self.model.load(file)
		self.state_dim = state_dim

	def forward(self, x, return_np=False):
		states = to_torch(x[:, :self.state_dim])
		actions = to_torch(x[:, self.state_dim:])
		dist = self.model.actor(states)
		actions = actions.clamp(*np.array([-1, 1])*0.999)
		log_prob = dist.log_prob(actions).sum(-1)-50

		if return_np:
			# batch
			return to_np(log_prob)
		else:
			# batch x 1
			return log_prob.unsqueeze(1)
