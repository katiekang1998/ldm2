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
from pendulum_env import PendulumEnv
from hopper_backwards import HopperBackwardsEnv

torch.cuda.set_device(0)
dataset_name = "expert"
ldm_path = "/home/katie/Desktop/ldm2/"
save_file = "data/flow_ldms_online/"+dataset_name+"/"
save_path = ldm_path+save_file
shutil.copytree(ldm_path, save_path+"training_files/", ignore=shutil.ignore_patterns("data"))


# #d4rl dataset
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

#learn density model
density_model = FlowDensityModel(ldm_path+"data/flows/"+dataset_name+"/flow.pt", state_dim, action_dim)
# # density_model.plot_samples(save_path)

#put data in replay buffer
rew = np.zeros((len_dataset))
for i in range(len_dataset//256):
  print(i)
  rew[i*256:(i+1)*256] = density_model(data[i*256:(i+1)*256, :state_dim+action_dim], return_np=True)
# rew[(len_dataset//256)*256:]=density_model(data[(len_dataset//256)*256:, :state_dim+action_dim], return_np=True)

rew = np.clip(rew, -200, 200)
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

print("Training LDM")
ldm = LDM(state_dim, action_dim, [-1, 1],
                 1e-4, [0.9, 0.999], 1e-4,
                 [0.9, 0.999], 0.005, 2,
                 1024, density_model=density_model)

# ldm = SACAgent(state_dim, action_dim, [-1, 1], 0.99,
#                1e-4, [0.9, 0.999], 1e-4,
#                [0.9, 0.999], 0.005, 2,
#                1024)



# pendulum = PendulumEnv()
env = HopperEnv()
done = True

save_every_n_steps = 2000
num_train_steps = 200000



with open(save_path+"rollout_length.csv", 'w+') as csvfile: 
  csvwriter = csv.writer(csvfile) 
  csvwriter.writerow(["step", "rollout_length"]) 


for step in range(num_train_steps+1):
  ldm.update(step, replay_buffer)

  if done:
    # obs = pendulum.reset()
    # obs = process_pendulum_obs(np.array([obs]))
    obs = env.reset()
    done = False
    episode_step = 0


  # sample action for data collection
  # import IPython; IPython.embed()

  action = ldm.act(obs, sample=True)

  next_obs, reward, done, failure = env.step(action)
  next_obs = next_obs.squeeze()

  # next_obs, _, _, _ = pendulum.step(action)
  # next_obs = process_pendulum_obs(np.array([next_obs])).squeeze(-1)

  done = done or episode_step > 1000
  reward = to_np(density_model(to_torch([np.concatenate([obs, action])])).squeeze())
  reward = np.clip(reward, -200, 200)
  replay_buffer.add(obs, action, reward, next_obs, False)

  obs = next_obs
  episode_step += 1
  
  if step%save_every_n_steps == 0:
    step_str = f"{step:07}"
    print(step_str)
    rollout_length, state_trajectories, action_trajectories, next_state_trajectories = run_closed_loop(ldm, num_episodes = 10)
    with open(save_path+"rollout_length.csv", 'a') as csvfile: 
      csvwriter = csv.writer(csvfile) 
      csvwriter.writerow([step_str, rollout_length]) 
    torch.save(ldm.actor.state_dict(), save_path+step_str+"actor.pt")
    torch.save(ldm.critic.state_dict(), save_path+step_str+"critic.pt")
    torch.save(ldm.critic_target.state_dict(), save_path+step_str+"critic_target.pt")
    # torch.save(ldm.log_alpha, save_path+step_str+"log_alpha.pt")
  step += 1



import IPython; IPython.embed()