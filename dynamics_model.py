import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class DynamicsModel(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(state_dim+action_dim, 128),
			nn.Tanh(),
			nn.Linear(128, 128),
			nn.Tanh(),
			nn.Linear(128, state_dim)).cuda()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
		self.loss_function = nn.MSELoss()
		self.batch_size = 256
		self.record_loss_per_n_steps = 50
		self.losses = []

	def update(self, step, replay_buffer):
		obs, action, reward, next_obs, not_done = replay_buffer.sample(
			self.batch_size)
		loss = self.loss_function(self.model(torch.cat([obs, action], 1)), next_obs)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		if step%self.record_loss_per_n_steps == 0:
			self.losses.append(loss.detach().cpu().numpy())

	def plot_loss(self):
		plt.plot([i*self.record_loss_per_n_steps for i in range(len(self.losses))], self.losses)
		plt.xlabel("Steps")
		plt.ylabel("Loss")
		plt.show()