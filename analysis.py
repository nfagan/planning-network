from model import AgentModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os

REPO_ROOT = '/Users/nick/Documents/mattarlab/project1/planning-network'

# -------------------------------------------------------------------------------------

def analysis_scalar(xs, ys, ylab):
  plt.figure(1)
  plt.clf()
  plt.plot(xs, ys)
  plt.ylabel(ylab)
  plt.show()

# -------------------------------------------------------------------------------------

if __name__ == '__main__':
  eval_p = os.path.join(REPO_ROOT, 'results', 'evaluation.pth')
  res = torch.load(eval_p)['rows']
  ri = range(len(res))

  p_plans = np.array([res[i]['res'].p_plan for i in ri])
  earned_rew = np.array([res[i]['res'].mean_total_reward for i in ri])
  reward_acc = np.array([res[i]['exploit_acc'] for i in ri])
  xs = np.array([res[i]['experience'] / 1e6 for i in ri])

  # analysis_scalar(xs, p_plans, 'p(plan)')
  # analysis_scalar(xs, earned_rew, 'mean reward')
  analysis_scalar(xs, reward_acc, 'exploit reward pred acc')