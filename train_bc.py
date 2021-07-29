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
from bc import BC

torch.cuda.set_device(1)
dataset_name = "expert"
ldm_path = "/home/katie/Desktop/ldm2/"
save_file = "data/bc/"+dataset_name+"/"
save_path = ldm_path+save_file
shutil.copytree(ldm_path, save_path+"training_files/", ignore=shutil.ignore_patterns("data"))


#d4rl dataset
hopper = gym.make('hopper-'+dataset_name+'-v2')
dataset = hopper.get_dataset()
data = np.concatenate([dataset['observations'], dataset['actions'], dataset['next_observations']], axis = 1)
state_dim = 11
action_dim = 3

# pkl_file = open('data/pendulum_sac_experts_action_space_1.pkl', 'rb')
# dataset = pickle.load(pkl_file)
# data = np.concatenate([process_pendulum_obs(dataset['observations']), dataset['actions'], process_pendulum_obs(dataset['next_observations'])], axis = 1)
# pkl_file.close()
# state_dim = 2
# action_dim = 1

len_dataset = len(data)


replay_buffer = ReplayBuffer([state_dim], [action_dim], len_dataset)
for i in range(len_dataset):
  replay_buffer.add(data[i][:state_dim], data[i][state_dim: state_dim+action_dim], 0, data[i][state_dim+action_dim:], False)

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

print("Training BC")
bc = BC(state_dim, action_dim, [-1, 1],
                 1e-4, [0.9, 0.999], 1024)

save_every_n_steps = 2000
num_train_steps = 100000
for step in range(num_train_steps+1):
  bc.update(step, replay_buffer)
  if step%save_every_n_steps == 0:
    step_str = f"{step:07}"
    rollout_length, state_trajectories, action_trajectories, next_state_trajectories = run_closed_loop(bc, num_episodes = 10)

    rollout_length = str(rollout_length)
    print(step_str)
    print(rollout_length)
    with open(save_path+"rollout_length.csv", 'a') as csvfile: 
      csvwriter = csv.writer(csvfile) 
      csvwriter.writerow([step_str, rollout_length]) 
    torch.save(bc.actor.state_dict(), save_path+step_str+"actor.pt")



# import IPython; IPython.embed()