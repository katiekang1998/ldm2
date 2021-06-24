from pendulum_env import PendulumEnv
from replay_buffer import ReplayBuffer
from min_actor_critic_ebm_init import SACAgent
import copy
import numpy as np
import torch
import torch.utils.data
from policy_learned_model import Policy
from dynamics_model import DynamicsModel
from ebm import EBM
import matplotlib.pyplot as plt
import pickle                                                                                                                                                                                                                                                                                                                                                                 
from gym.envs.mujoco import HopperEnv
import gym
from utils import plot_hopper, process_pendulum_obs, plot_pendulum_ebm, plot_pendulum_ldm, plot_pendulum_ebm2, plot_pendulum_ldm2
import os
import shutil
import pickle
from utils import MLP

torch.cuda.set_device(1)
ldm_path = "/home/katie/Desktop/ldm/"
save_file = "data/cartpole_ldm_goal_ebm_init_no_entropy2_actor_update_only/"
save_path = ldm_path+save_file
# os.mkdir(save_path)

shutil.copytree(ldm_path, save_path+"training_files/", ignore=shutil.ignore_patterns("data"))

state_dim = 4
action_dim = 1

 

loadpaths = ['data/cartpole_sac_threshold45_goal_1.pkl',
'data/cartpole_sac_threshold45_goal_neg1.pkl',
'data/cartpole_sac_threshold45_goal_0pt5.pkl',
'data/cartpole_sac_threshold45_goal_neg0pt5.pkl',
'data/cartpole_sac_threshold45_goal_0.pkl',
'data/cartpole_sac_threshold45_goal_none.pkl']

data = load_buffer(loadpaths)
np.random.shuffle(data)
train_idx = int(0.9*len(data))

ebm = MLP(
    input_dim=state_dim+action_dim,
    hidden_dims=[128, 128, 128],
    output_dim=1,
).cuda()

ebm.load_state_dict(torch.load("data/cartpole_goal_ebm2.pt"))
# import IPython; IPython.embed()

# plot_pendulum_ebm(ebm, 1, save_path)

torch.save(ebm.state_dict(), save_path+"ebm.pt")

#Put data in replay buffer
ebm_rew = np.zeros((train_idx))
for i in range(train_idx//256):
  ebm_rew[i*256:(i+1)*256] = ebm.forward(torch.from_numpy(data[i*256:(i+1)*256, :state_dim+action_dim]).type(torch.FloatTensor).cuda()).detach().cpu().numpy().squeeze()

ebm_rew[(train_idx//256)*256:]=ebm.forward(torch.from_numpy(data[(train_idx//256)*256:train_idx, :state_dim+action_dim]).type(torch.FloatTensor).cuda()).detach().cpu().numpy().squeeze()

replay_buffer = ReplayBuffer([state_dim], [action_dim], train_idx)

for i in range(train_idx):
  # Fitted distribution
  replay_buffer.add(data[i][:state_dim], data[i][state_dim: state_dim+action_dim], ebm_rew[i], data[i][state_dim+action_dim:], False)


#Train dynamics model
print("Training model")
model = DynamicsModel(state_dim, action_dim)
for step in range(10000):
  model.update(step, replay_buffer)
torch.save(model.state_dict(), save_path+"dynamics_model.pt")


# #Train LDM
print("Training LDM")
zero_entropy =True
ldm = SACAgent(state_dim, action_dim, [-1, 1], 0.99,
                 1e-4, [0.9, 0.999], 1e-4,
                 [0.9, 0.999], 0.005, 2,
                 1024, zero_entropy=zero_entropy, ebm = ebm, q_init=ebm)
save_every_n_steps = 2000
num_train_steps = 100000
for step in range(num_train_steps+1):
  ldm.update_actor_only(step, replay_buffer)
  if step%save_every_n_steps == 0:
    step_str = f"{step:07}"
    print(step_str)
    torch.save(ldm.actor.state_dict(), save_path+step_str+"actor.pt")
    torch.save(ldm.critic.state_dict(), save_path+step_str+"critic.pt")
    torch.save(ldm.critic_target.state_dict(), save_path+step_str+"critic_target.pt")
    if not zero_entropy:
      torch.save(ldm.log_alpha, save_path+step_str+"log_alpha.pt")

step_str = f"{step:07}"
print(step_str)
torch.save(ldm.actor.state_dict(), save_path+step_str+"actor.pt")
torch.save(ldm.critic.state_dict(), save_path+step_str+"critic.pt")
torch.save(ldm.critic_target.state_dict(), save_path+step_str+"critic_target.pt")
if not zero_entropy:
  torch.save(ldm.log_alpha, save_path+step_str+"log_alpha.pt")


import IPython; IPython.embed()