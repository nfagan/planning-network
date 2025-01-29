from model import AgentModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os

def analysis_scalar(xs, ys, ylab):
  f = plt.figure(1)
  plt.clf()
  plt.plot(xs, ys)
  plt.ylabel(ylab)
  plt.show()
  plt.draw()
  f.savefig(os.path.join(os.getcwd(), 'plots', f'{ylab}.png'))

if __name__ == '__main__':
  eval_p = os.path.join(os.getcwd(), 'results', 'evaluation.pth')
  res = torch.load(eval_p)['rows']
  ri = range(len(res))
  # rv = 'res'
  rv = 'train_res'

  p_plans = np.array([res[i][rv].p_plan for i in ri])
  earned_rew = np.array([res[i][rv].mean_total_reward for i in ri])
  reward_acc = np.array([res[i]['exploit_acc'] for i in ri])
  state_acc = np.array([res[i][rv].state_prediction_acc for i in ri])
  xs = np.array([res[i]['experience'] / 1e6 for i in ri])

  analysis_scalar(xs, p_plans, 'p(plan)')
  analysis_scalar(xs, earned_rew, 'mean reward')
  analysis_scalar(xs, reward_acc, 'exploit reward pred acc')
  analysis_scalar(xs, state_acc, 'state pred acc')