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

from nflib.flows import (
	AffineConstantFlow, ActNorm, AffineHalfFlow, 
	SlowMAF, MAF, IAF, Invertible1x1Conv,
	NormalizingFlow, NormalizingFlowModel,
)
from nflib.spline_flows import NSF_AR, NSF_CL


class FlowDensityModel(nn.Module):
	def __init__(self, file, state_dim, action_dim):
		super(FlowDensityModel, self).__init__()
		prior = TransformedDistribution(Uniform(torch.zeros(state_dim+action_dim).cuda(), torch.ones(state_dim+action_dim).cuda()), SigmoidTransform().inv) # Logistic distribution

		# Neural splines, coupling
		nfs_flow = NSF_CL if True else NSF_AR
		flows = [nfs_flow(dim=state_dim+action_dim, K=64, B=20, hidden_dim=256) for _ in range(8)]
		convs = [Invertible1x1Conv(dim=state_dim+action_dim) for _ in flows]
		norms = [ActNorm(dim=state_dim+action_dim) for _ in flows]
		flows = list(itertools.chain(*zip(norms, convs, flows)))

		# construct the model
		self.model = NormalizingFlowModel(prior, flows).cuda()
		self.model.load_state_dict(torch.load(file))
		self.state_dim = state_dim

	def forward(self, x, return_np=False):
		zs, prior_logprob, log_det = self.model(to_torch(x))
		logprob = prior_logprob + log_det

		if return_np:
			# batch
			return to_np(logprob)
		else:
			# batch x 1
			return logprob.unsqueeze(1)

	def plot_samples(self, save_path):
		samples = self.model.sample(256)[-1].detach().cpu().numpy()
		plot_samples(save_path + "flow_samples.png", samples[:, :self.state_dim])
