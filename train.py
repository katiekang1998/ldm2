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

torch.cuda.set_device(0)
ldm_path = "/home/katie/Desktop/ldm2/"
save_file = "data/delete/"
save_path = ldm_path+save_file
shutil.copytree(ldm_path, save_path+"training_files/", ignore=shutil.ignore_patterns("data"))

state_dim = 11
action_dim = 3
 

#d4rl dataset
hopper = gym.make('hopper-medium-replay-v2')
dataset = hopper.get_dataset()
data = np.concatenate([dataset['observations'], dataset['actions'], dataset['next_observations']], axis = 1)
len_dataset = len(data)


#learn density model
density_model = FlowDensityModel(ldm_path+"data/flows/medium-replay/flow.pt", state_dim, action_dim)
density_model.plot_samples(save_path)
# import IPython; IPython.embed()

#put data in replay buffer
rew = np.zeros((len_dataset))
for i in range(len_dataset//256):
  rew[i*256:(i+1)*256] = density_model(data[i*256:(i+1)*256, :state_dim+action_dim], return_np=True)
rew[(len_dataset//256)*256:]=density_model(data[(len_dataset//256)*256:, :state_dim+action_dim], return_np=True)

rew = np.clip(rew, -200, 0)
replay_buffer = ReplayBuffer([state_dim], [action_dim], len_dataset)

for i in range(len_dataset):
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
  csvwriter.writerow(["step", "rollout_length"]) 

  print("Training LDM")
  ldm = LDM(state_dim, action_dim, [-1, 1],
                   1e-4, [0.9, 0.999], 1e-4,
                   [0.9, 0.999], 0.005, 2,
                   1024, density_model=density_model)
  save_every_n_steps = 2000
  num_train_steps = 100000
  for step in range(num_train_steps+1):
    ldm.update(step, replay_buffer)
    if step%save_every_n_steps == 0:
      step_str = f"{step:07}"
      rollout_length = str(run_closed_loop(ldm))
      print(step_str)
      print(rollout_length)
      csvwriter.writerow([step_str, rollout_length]) 
      torch.save(ldm.actor.state_dict(), save_path+step_str+"actor.pt")
      torch.save(ldm.critic.state_dict(), save_path+step_str+"critic.pt")
      torch.save(ldm.critic_target.state_dict(), save_path+step_str+"critic_target.pt")


# import IPython; IPython.embed()