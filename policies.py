import torch
import numpy as np
import copy

class ThresholdPolicy():
  def __init__(self, action_space, ldm, num_random_actions = 1000, threshold=-6):
    self.action_space = action_space
    self.num_random_actions = num_random_actions
    self.ldm = ldm
    self.threshold = threshold

  def get_action(self, state):

    random_actions = np.random.uniform(self.action_space.low, self.action_space.high, size = (self.num_random_actions, len(self.action_space.low)))
    random_actions_batch = torch.from_numpy(random_actions).type(torch.FloatTensor).cuda()
    state_batch = torch.from_numpy(np.array(state)).type(torch.FloatTensor).cuda().repeat(self.num_random_actions, 1)

    Q1s, Q2s = self.ldm.critic(state_batch, random_actions_batch)
    indices_above_threshold = (torch.min(Q1s, Q2s).squeeze()>=self.threshold).nonzero().flatten()
    if len(indices_above_threshold)>0:
      print("Random action")
      action = np.array(random_actions[np.random.choice(indices_above_threshold.cpu().detach().numpy())])
      random_action = True
    else:
      print("Safety action")
      action = self.ldm.act(state)
      random_action = False
      # raise Exception("no actions found")
    return action