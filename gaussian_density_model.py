from scipy.stats import multivariate_normal
import torch.nn as nn
import numpy as np
from utils import *

class GaussianDensityModel(nn.Module):
	def __init__(self, data, state_dim, action_dim):
		super(GaussianDensityModel, self).__init__()
		mean = np.mean(data[:,:state_dim+action_dim], 0)
		cov = np.cov(data[:,:state_dim+action_dim], rowvar=0)
		self.state_dim = state_dim
		self.distribution = multivariate_normal(mean, cov)

	def forward(self, x, return_np=False):
		densities_np = np.log(self.distribution.pdf(to_np(x)))-10
		if return_np:
			return densities_np
		else:
			return to_torch(densities_np).unsqueeze(1)

	def plot_samples(self, save_path):
		plot_samples(save_path + "multivariate_gaussian_samples.png", self.distribution.rvs(size=256)[:, :self.state_dim])
