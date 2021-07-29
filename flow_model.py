from utils import *
import csv

import itertools

import numpy as np
import matplotlib.pyplot as plt

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

import gym
import d4rl
import os
import shutil

logdir = "/home/katie/Desktop/ldm2/data"

torch.cuda.set_device(1)
ldm_path = "/home/katie/Desktop/ldm2/"
save_file = "data/flows/random/"
save_path = ldm_path+save_file
shutil.copytree(ldm_path, save_path+"training_files/", ignore=shutil.ignore_patterns("data"))


def plot_scatter(savepath, samples, plotting_index, xlim=None, ylim=None):
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


#d4rl dataset
hopper = gym.make('hopper-random-v2')
dataset = hopper.get_dataset()
data = np.concatenate([dataset['observations'], dataset['actions']], axis = 1)

# #pendulum dataset
# pkl_file = open('data/pendulum_sac_experts_action_space_1.pkl', 'rb')
# dataset = pickle.load(pkl_file)
# data = np.concatenate([process_pendulum_obs(dataset['observations']), dataset['actions']], axis = 1)
# pkl_file.close()

len_dataset = len(data)
np.random.shuffle(data)

#get plotting indices
cov = np.cov(data, rowvar=False)
plotting_indices = np.array(list(zip([_ for _ in range((data).shape[1])], np.argmax(np.argsort(abs(cov)), axis=1))))
# plotting_indices = [[0, 1]]

for i in range(len(plotting_indices)):
	plot_scatter(os.path.join(save_path, str(i)+'samples_data.png'), data[:10000], plotting_indices[i])
	plot_hist(os.path.join(save_path, str(i)+'hist_data.png'), data[:10000], plotting_indices[i])

plot_samples(os.path.join(save_path, 'hoppers_data.png'), data[:256,:11])

prior = TransformedDistribution(Uniform(torch.zeros(len(data[0])).cuda(), torch.ones(len(data[0])).cuda()), SigmoidTransform().inv) # Logistic distribution

# Neural splines, coupling
nfs_flow = NSF_CL if True else NSF_AR
flows = [nfs_flow(dim=len(data[0]), K=64, B=20, hidden_dim=256) for _ in range(8)]
convs = [Invertible1x1Conv(dim=len(data[0])) for _ in flows]
norms = [ActNorm(dim=len(data[0])) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

# construct the model
model = NormalizingFlowModel(prior, flows).cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

model.train()
# for epoch in range(8):
for k in range(300):
	x = torch.from_numpy(data[k*1028:(k+1)*1028].astype(np.float32)).cuda()

	zs, prior_logprob, log_det = model(x)
	logprob = prior_logprob + log_det
	loss = torch.clamp(-torch.mean(logprob), -2000, 2000) # NLL

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	print(loss.item())

	if torch.isnan(loss):
		import IPython; IPython.embed()

	# if k%100==0:
	# 	samples = model.sample(1000)[-1].detach().cpu().numpy()
	# 	for i in range(len(plotting_indices)):
	# 		plot_scatter(os.path.join(save_path, str(i)+'samples_flow.png'), samples[:1000], plotting_indices[i])
	# 		plot_hist(os.path.join(save_path, str(i)+'hist_flow.png'), samples[:1000], plotting_indices[i])
	# 	plot_samples(os.path.join(save_path, f'hoppers{k:05}.png'), np.array(samples)[:256,:11])



torch.save(model.state_dict(), save_path+"flow.pt")

samples = model.sample(1000)[-1].detach().cpu().numpy()


for i in range(len(plotting_indices)):
	plot_scatter(os.path.join(save_path, str(i)+'samples_flow.png'), samples[:1000], plotting_indices[i])
	plot_hist(os.path.join(save_path, str(i)+'hist_flow.png'), samples[:1000], plotting_indices[i])


plot_samples(os.path.join(save_path, 'hoppers.png'), np.array(samples)[:256,:11])