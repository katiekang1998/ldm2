from ldm import LDM
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from gym.envs.mujoco import HopperEnv
import gym
import os
import time

def run_closed_loop(ldm, len_episode=1000, num_episodes = 1, render = False):
	env = HopperEnv()
	steps = []
	for i in range(num_episodes):
		next_obs = env.reset()
		if render:
			env.render()
		done = False
		step = 0
		while((not done or render) and step<len_episode):
			obs = next_obs
			action = ldm.act(obs, sample=False)
			print(action)
			next_obs, reward, done, failure = env.step(action)
			time.sleep(.002)
			if render:
				env.render()
			step +=1
		steps.append(step)
	return np.array(steps).mean()


if __name__ == "__main__":
	torch.cuda.set_device(0)
	save_path = "/home/katie/Desktop/ldm2/data/medium_replay2/"
	state_dim = 11
	action_dim =3
	step_str = f"{98000:07}"

	ldm = LDM(state_dim, action_dim, [-1, 1],
	                 1e-4, [0.9, 0.999], 1e-4,
	                 [0.9, 0.999], 0.005, 2,
	                 1024)

	ldm.load(save_path+step_str+"actor.pt", save_path+step_str+"critic.pt", save_path+step_str+"critic_target.pt")

	run_closed_loop(ldm, render=True)