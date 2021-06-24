import torch
import numpy as np
import copy

class Policy():
  def __init__(self, dynamics_model, action_space, ldm, num_random_actions = 1000, threshold=-6):
    self.dynamics_model = dynamics_model
    self.action_space = action_space
    self.num_random_actions = num_random_actions
    self.ldm = ldm
    self.threshold = threshold

  def get_action(self, state, safety_action_only = False, ebm=None, goal = None, no_constraint = False):
    if no_constraint:
      print("here")
      random_actions = np.random.uniform(self.action_space[0], self.action_space[1], size = (self.num_random_actions, len(self.action_space[1])))
      random_actions_batch = torch.from_numpy(random_actions).type(torch.FloatTensor).cuda()
      state_batch = torch.from_numpy(np.array(state)).type(torch.FloatTensor).cuda().repeat(self.num_random_actions, 1)
      state_actions_batch = torch.cat([state_batch, random_actions_batch], dim=1)
      next_states_batch = self.dynamics_model(state_actions_batch)
      best_idx = torch.argmin((next_states_batch[:, 0]+goal)**2).cpu().detach().numpy()
      action = random_actions_batch[best_idx].cpu().detach().numpy()
      random_action = True
      return action, random_action
    # print(state)
    if safety_action_only:
      # print("Safety action")
      action = self.ldm.act(state)
      random_action = False

      return action, random_action
    else:
      random_actions = np.random.uniform(self.action_space[0], self.action_space[1], size = (self.num_random_actions, len(self.action_space[1])))
      random_actions_batch = torch.from_numpy(random_actions).type(torch.FloatTensor).cuda()
      state_batch = torch.from_numpy(np.array(state)).type(torch.FloatTensor).cuda().repeat(self.num_random_actions, 1)

      Q1s, Q2s = self.ldm.critic(state_batch, random_actions_batch)
      indices_above_threshold = (torch.min(Q1s, Q2s).squeeze()>=self.threshold).nonzero().flatten()
      if len(indices_above_threshold)>0:
        if not goal:
          # print("Random action")
          action = np.array(random_actions[np.random.choice(indices_above_threshold.cpu().detach().numpy())])
          random_action = True
        else:
          print("Goal action")
          actions_filtered_batch = torch.from_numpy(np.array(random_actions[indices_above_threshold.cpu().detach().numpy()])).type(torch.FloatTensor).cuda()
          states_filtered_batch = torch.from_numpy(np.array(state)).type(torch.FloatTensor).cuda().repeat(len(actions_filtered_batch), 1)
          state_actions_filtered_batch = torch.cat([states_filtered_batch, actions_filtered_batch], dim=1)
          next_states_batch = self.dynamics_model(state_actions_filtered_batch)
          best_idx = torch.argmin((next_states_batch[:, 0]-goal)**2).cpu().detach().numpy()
          action = actions_filtered_batch[best_idx].cpu().detach().numpy()
          random_action = True
      else:
        print("Safety action")
        action = self.ldm.act(state)
        random_action = False
        # raise Exception("no actions found")

      return action, random_action