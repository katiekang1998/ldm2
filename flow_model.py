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

torch.cuda.set_device(0)
ldm_path = "/home/katie/Desktop/ldm2/"
save_file = "data/flows/forwards-backwards-replay/"
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
hopper = gym.make('hopper-medium-replay-v2')
dataset = hopper.get_dataset()
data_orig = np.concatenate([dataset['observations'], dataset['actions']], axis = 1)
np.random.shuffle(data_orig)
data = data_orig[:190000]
holdout_data = data_orig[190000:200000]
print(holdout_data.shape)

pkl_file = open(ldm_path+'data/hopper_backwards_replay.pkl', 'rb')
dataset = pickle.load(pkl_file)
data2 = np.concatenate([dataset['observations'], dataset['actions']], axis = 1)
pkl_file.close()

data = np.concatenate([data, data2[:190000]])
holdout_data = np.concatenate([holdout_data, data2[190000:]])
print(holdout_data.shape)
np.random.shuffle(data)
np.random.shuffle(holdout_data)
import IPython; IPython.embed()

#get plotting indices
cov = np.cov(data, rowvar=False)
plotting_indices = np.array(list(zip([_ for _ in range((data).shape[1])], np.argmax(np.argsort(abs(cov)), axis=1))))

for i in range(len(plotting_indices)):
	plot_scatter(os.path.join(save_path, str(i)+'samples_data.png'), data[:10000], plotting_indices[i])
	plot_hist(os.path.join(save_path, str(i)+'hist_data.png'), data[:10000], plotting_indices[i])

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
for epoch in range(2):
	for k in range(len(data)//128):
		x = torch.from_numpy(data[k*128:(k+1)*128].astype(np.float32)).cuda()

		zs, prior_logprob, log_det = model(x)
		logprob = prior_logprob + log_det
		loss = -torch.sum(logprob) # NLL

		model.zero_grad()
		loss.backward()
		optimizer.step()

		if k % 128 == 0:
			print("loss")
			print(loss.item())
			x = torch.from_numpy(holdout_data[k%10000:k%10000+128].astype(np.float32)).cuda()
			zs, prior_logprob, log_det = model(x)
			logprob = prior_logprob + log_det
			loss = -torch.sum(logprob) # NLL
			print("holdout loss")
			print(loss.item())

torch.save(model.state_dict(), save_path+"flow.pt")

samples = model.sample(1000)[-1].detach().cpu().numpy()


for i in range(len(plotting_indices)):
	plot_scatter(os.path.join(save_path, str(i)+'samples_flow.png'), samples[:1000], plotting_indices[i])
	plot_hist(os.path.join(save_path, str(i)+'hist_flow.png'), samples[:1000], plotting_indices[i])


plot_samples(os.path.join(save_path, 'samples.png'), np.array(samples)[:256,:11])