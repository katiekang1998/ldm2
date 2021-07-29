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


def run_closed_loop(ldm, len_episode=1000, num_episodes = 1, render = False, env = HopperEnv()):
	state_trajectories = []
	action_trajectories = []
	next_state_trajectories = []
	len_trajectories = []
	for i in range(num_episodes):
		next_obs = env.reset()
		if isinstance(env, PendulumEnv):
			next_obs = process_pendulum_obs(np.array([next_obs]))
			print(next_obs)
		if render:
			env.render()
		done = False
		dones = []
		states = []
		actions = []
		next_states = []
		step = 0
		while(step<len_episode): #or render) and step<len_episode):
			obs = next_obs
			action = ldm.act(obs, sample=False)
			next_obs, reward, done, failure = env.step(action)
			next_obs = next_obs.squeeze()
			if isinstance(env, PendulumEnv)	:
				next_obs = process_pendulum_obs(np.array([next_obs]))
				print(next_obs)
			if render:
				env.render()				
			step +=1
			states.append(obs.squeeze())
			actions.append(action)
			next_states.append(next_obs.squeeze())
			dones.append(done)
		state_trajectories.append(states)
		action_trajectories.append(actions)
		next_state_trajectories.append(next_states)
		len_trajectories.append(len_episode - np.sum(dones))
	#.reshape(num_episodes, -1, env.action_space.shape[0])

	return len_trajectories, np.array(state_trajectories), np.array(action_trajectories), np.array(next_state_trajectories)


def evaluate_experiment(ldm_name, density_model_name):
	save_file = "data/flow_ldms/"+ldm_name+"/"
	save_path = ldm_path+save_file
	state_dim = 11
	action_dim =3
	density_model = FlowDensityModel(ldm_path+"data/flows/"+density_model_name+"/flow.pt", state_dim, action_dim)
	hopper = HopperEnv()

	ldm = LDM(state_dim, action_dim, [-1, 1],
					 1e-4, [0.9, 0.999], 1e-4,
					 [0.9, 0.999], 0.005, 2,
					 1024)
	
	mean_len_traj_itr = []
	mean_densities_itr = []
	for j in range(50):
		step_str = f"{2000*j:07}"
		print(step_str)
		ldm.load(save_path+step_str+"actor.pt", save_path+step_str+"critic.pt", save_path+step_str+"critic_target.pt")#, save_path+step_str+"log_alpha.pt")

		len_trajectories, state_trajectories, action_trajectories, next_state_trajectories = run_closed_loop(ldm, len_episode=1000, render=False, env=hopper, num_episodes = 5)
		mean_densities = []
		for i in range(len(state_trajectories)):
			densities = np.clip(density_model(np.concatenate([state_trajectories[i], action_trajectories[i]], 1), return_np=True), -200, 200)
			mean_densities.append(np.mean(densities))

		mean_len_traj_itr.append(np.mean(len_trajectories))
		mean_densities_itr.append(np.mean(mean_densities))

	return mean_len_traj_itr, mean_densities_itr


torch.cuda.set_device(0)
ldm_path = "/home/katie/Desktop/ldm2/"

mean_len_traj_itr_0, mean_densities_itr_0 = evaluate_experiment("expert", "expert")
mean_len_traj_itr_0pt1, mean_densities_itr_0pt1 = evaluate_experiment("expert_cql_0.1", "expert")
mean_len_traj_itr_1, mean_densities_itr_1 = evaluate_experiment("expert_cql_1", "expert")
mean_len_traj_itr_10, mean_densities_itr_10 = evaluate_experiment("expert_cql_10", "expert")

import IPython; IPython.embed()

idxs = [i for i in range(50)] 
plt.plot(idxs, mean_len_traj_itr_0, label = "cql coefficient 0") 
plt.plot(idxs, mean_len_traj_itr_0pt1, label = "cql coefficient 0.1") 
plt.plot(idxs, mean_len_traj_itr_1, label = "cql coefficient 1") 
plt.plot(idxs, mean_len_traj_itr_10, label = "cql coefficient 10") 
plt.legend()
plt.xlabel("step/2000")
plt.ylabel("avg traj len")
plt.show()


plt.plot(idxs, mean_densities_itr_0, label = "cql coefficient 0") 
plt.plot(idxs, mean_densities_itr_0pt1, label = "cql coefficient 0.1") 
plt.plot(idxs, mean_densities_itr_1, label = "cql coefficient 1") 
plt.plot(idxs, mean_densities_itr_10, label = "cql coefficient 10") 
plt.legend()
plt.xlabel("step/2000")
plt.ylabel("avg rollout density")
plt.show()