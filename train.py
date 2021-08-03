from replay_buffer import ReplayBuffer
from ldm import LDM
import copy
import numpy as np
import torch
from dynamics_model import DynamicsModel
import matplotlib.pyplot as plt
import pickle                                                                                                                                                                                                                                                                                                                                                                 
from gym.envs.mujoco import HopperEnv
import gym
import os
import shutil
import pickle
import d4rl
from gaussian_density_model import GaussianDensityModel
from flow_density_model import FlowDensityModel
from test import run_closed_loop
from utils import *
import csv 
from sac import SACAgent
from bc_density_model import BCDensityModel

torch.cuda.set_device(0)
dataset_name = "medium-expert"
ldm_path = "/home/katie/Desktop/ldm2/"
save_file = "data/cql/"+dataset_name+"_cql2_lagrange_density_rew/"
save_path = ldm_path+save_file
shutil.copytree(ldm_path, save_path+"training_files/", ignore=shutil.ignore_patterns("data"))


#d4rl dataset
hopper = gym.make('hopper-'+dataset_name+'-v2')
dataset = hopper.get_dataset()
data = np.concatenate([dataset['observations'], dataset['actions'], dataset['next_observations']], axis = 1)
dones = dataset['terminals']
state_dim = 11
action_dim = 3


# pkl_file = open('data/pendulum_sac_experts_action_space_1.pkl', 'rb')
# dataset = pickle.load(pkl_file)
# data = np.concatenate([process_pendulum_obs(dataset['observations']), dataset['actions'], process_pendulum_obs(dataset['next_observations'])], axis = 1)
# pkl_file.close()
# state_dim = 2
# action_dim = 1

len_dataset = len(data)

#learn density model
density_model = FlowDensityModel(ldm_path+"data/flows/"+dataset_name+"/flow.pt", state_dim, action_dim)
# density_model = BCDensityModel(ldm_path+"data/bc/"+dataset_name+"/0100000actor.pt", state_dim, action_dim)
# density_model.plot_samples(save_path)

#put data in replay buffer
rew = np.zeros((len_dataset))
for i in range(len_dataset//1024):
  rew[i*1024:(i+1)*1024] = density_model(data[i*1024:(i+1)*1024, :state_dim+action_dim], return_np=True)
rew[(len_dataset//1024)*1024:]=density_model(data[(len_dataset//1024)*1024:, :state_dim+action_dim], return_np=True)
# rew = np.clip(rew, -200, 200)
# min_density = min(rew)
# std_density = rew.std()
# threshold = np.percentile(rew, 60)
# rew = rew > threshold


replay_buffer = ReplayBuffer([state_dim], [action_dim], len_dataset)

for i in range(len_dataset):
  # if dones[i]:
  #   rew[i] = min_density - std_density
  replay_buffer.add(data[i][:state_dim], data[i][state_dim: state_dim+action_dim], rew[i], data[i][state_dim+action_dim:], False)

# #Train dynamics model
# print("Training model")
# model = DynamicsModel(state_dim, action_dim)
# for step in range(10000):
#   model.update(step, replay_buffer)
# torch.save(model.state_dict(), save_path+"dynamics_model.pt")

# #Train LDM
with open(save_path+"rollout_length.csv", 'w+') as csvfile: 
  csvwriter = csv.writer(csvfile) 
  csvwriter.writerow(["step", "rollout_length", "density_mean"]) 

print("Training LDM")
# ldm = LDM(state_dim, action_dim, [-1, 1],
#                  1e-4, [0.9, 0.999], 1e-4,
#                  [0.9, 0.999], 0.005, 2,
#                  1024, density_model=density_model, cql_regularizer_coefficient=1)

ldm = SACAgent(state_dim, action_dim, [-1, 1], 0.99,
               1e-4, [0.9, 0.999], 1e-4,
               [0.9, 0.999], 0.005, 2,
               1024, with_lagrange=True)

# step_str = f"{98000:07}"
# ldm.load(save_path+step_str+"actor.pt", save_path+step_str+"critic.pt", save_path+step_str+"critic_target.pt", save_path+step_str+"log_alpha.pt")

save_every_n_steps = 2000
num_train_steps = 100000
for step in range(num_train_steps+1):
  ldm.update(step, replay_buffer)
  if step%save_every_n_steps == 0:
    step_str = f"{step:07}"
    rollout_length, state_trajectories, action_trajectories, next_state_trajectories = run_closed_loop(ldm, num_episodes = 10)

    density_means = []
    for i in range(len(state_trajectories)):
      density_means.append(np.clip(density_model(np.concatenate([state_trajectories[i], action_trajectories[i]], 1), return_np=True), -200, 200).mean())
    mean_density_means = np.mean(density_means)

    rollout_length = str(rollout_length)
    mean_density_means = str(mean_density_means)
    print(step_str)
    print(rollout_length)
    print(mean_density_means)
    with open(save_path+"rollout_length.csv", 'a') as csvfile: 
      csvwriter = csv.writer(csvfile) 
      csvwriter.writerow([step_str, rollout_length, mean_density_means]) 
    torch.save(ldm.actor.state_dict(), save_path+step_str+"actor.pt")
    torch.save(ldm.critic.state_dict(), save_path+step_str+"critic.pt")
    torch.save(ldm.critic_target.state_dict(), save_path+step_str+"critic_target.pt")
    torch.save(ldm.log_alpha, save_path+step_str+"log_alpha.pt")



# import IPython; IPython.embed()