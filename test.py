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
from flow_density_model import FlowDensityModel
from utils import *
from sac import SACAgent
from pendulum_env import PendulumEnv
from policies import ThresholdPolicy
from bc import BC

def run_closed_loop(ldm, len_episode=1000, num_episodes = 1, render = False, env = HopperEnv(), policy = None):
	steps = []
	state_trajectories = []
	action_trajectories = []
	next_state_trajectories = []
	for i in range(num_episodes):
		next_obs = env.reset()
		if isinstance(env, PendulumEnv):
			next_obs = process_pendulum_obs(np.array([next_obs]))
			print(next_obs)
		if render:
			env.render()
		done = False
		step = 0
		states = []
		actions = []
		next_states = []
		while((not done or render) and step<len_episode): #or render) and step<len_episode):
			obs = next_obs
			if policy:
				action = policy.get_action(obs)
			else:
				action = ldm.act(obs, sample=False)
			# print(action)
			next_obs, reward, done, failure = env.step(action)
			next_obs = next_obs.squeeze()
			if isinstance(env, PendulumEnv)	:
				next_obs = process_pendulum_obs(np.array([next_obs]))
				print(next_obs)
			# time.sleep(.002)
			if render:
				env.render()				
			step +=1
			states.append(obs.squeeze())
			actions.append(action)
			next_states.append(next_obs.squeeze())
		steps.append(step)
		state_trajectories.append(states)
		action_trajectories.append(actions)
		next_state_trajectories.append(next_states)
	print(steps)
	#.reshape(num_episodes, -1, env.action_space.shape[0])

	return np.array(steps).mean(), np.array(state_trajectories), np.array(action_trajectories), np.array(next_state_trajectories)

if __name__ == "__main__":
	torch.cuda.set_device(0)
	ldm_path = "/home/katie/Desktop/ldm2/"
	experiment = "expert"
	save_file = "data/bc/"+experiment+"/"
	save_path = ldm_path+save_file
	state_dim = 11
	action_dim =3

	# state_dim = 2
	# action_dim =1

	step_str = f"{100000:07}"

	# ldm = LDM(state_dim, action_dim, [-1, 1],
	#                  1e-4, [0.9, 0.999], 1e-4,
	#                  [0.9, 0.999], 0.005, 2,
	#                  1024)

	# ldm = SACAgent(state_dim, action_dim, [-1, 1], 0.99,
 #         1e-4, [0.9, 0.999], 1e-4,
 #         [0.9, 0.999], 0.005, 2,
 #         1024)

	bc = BC(state_dim, action_dim, [-1, 1], 1e-4, [0.9, 0.999], 1024)
	
	# ldm.load(save_path+step_str+"actor.pt", save_path+step_str+"critic.pt", save_path+step_str+"critic_target.pt")#, save_path+step_str+"log_alpha.pt")
	bc.load(save_path+step_str+"actor.pt")

	# policy = ThresholdPolicy(HopperEnv().action_space, ldm, num_random_actions = 1000, threshold=-45)

	# import IPython; IPython.embed()
	rollout_length_mean, state_trajectories, action_trajectories, next_state_trajectories = run_closed_loop(bc, len_episode=1000, render=True, env=HopperEnv(), num_episodes = 10) #, policy=policy)
	
	for i in range(len(state_trajectories)):
		density_model = FlowDensityModel(ldm_path+"data/flows/"+experiment+"/flow.pt", state_dim, action_dim)
		densities = np.clip(density_model(np.concatenate([state_trajectories[i], action_trajectories[i]], 1), return_np=True), -200, 200)
		print(np.mean(densities))

		# q1, q2 = ldm.critic(to_torch(state_trajectories[i]), to_torch(action_trajectories[i]))
		# ldm_values = to_np(torch.min(q1, q2)).squeeze()

		# idx = [i for i in range(len(ldm_values))]
		# plt.plot(idx, ldm_values, label = "ldm")
		# plt.plot(idx, densities, label = "densities") 
		# plt.legend()
		# plt.xlabel("trajectory step")
		# plt.show()

		# td_error = ldm.get_td_error(state_trajectories[i], action_trajectories[i], densities, next_state_trajectories[i]) 
		# plt.plot(idx, td_error, label = "td error")
		# plt.legend()
		# plt.xlabel("trajectory step")
		# plt.show()

