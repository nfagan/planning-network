import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import os
import glob

ROOT_P = os.getcwd()

def load_results():
  eval_p = os.path.join(ROOT_P, 'results')
  eval_fs = glob.glob(f'{eval_p}/*.pth')
  res = [torch.load(x)['row'] for x in eval_fs]
  res = sorted(res, key=lambda x: x['experience'])
  return res

def analysis_scalar(xs, ys, ylab, save_p, ylim=None):
  f = plt.figure(1)
  plt.clf()
  plt.plot(xs, ys)
  plt.ylabel(ylab)
  if ylim is not None: plt.ylim(ylim)
  plt.show()
  plt.draw()
  if save_p is not None:
    f.savefig(os.path.join(save_p, f'{ylab}.png'))

def analysis_multi(xs, zs, by, ylab, save_p, ylim=None):
  f = plt.figure(1)
  plt.clf()
  h = plt.plot(xs, zs)
  plt.ylabel(ylab)
  num_lines = len(by)
  cmap = plt.get_cmap('spring', num_lines)(np.arange(num_lines))[:, :3]
  for i in range(num_lines):
    h[i].set_color(cmap[i, :])
    h[i].set_label(f'{by[i]}')
  if ylim is not None: plt.ylim(ylim)
  # import pdb; pdb.set_trace()
  plt.legend()
  plt.show()
  plt.draw()
  if save_p is not None:
    f.savefig(os.path.join(save_p, f'{ylab}.png'))

if __name__ == '__main__':
  subdir = '60'
  save_p = os.path.join(ROOT_P, 'plots', subdir)
  if save_p is not None: os.makedirs(save_p, exist_ok=True)

  res = load_results()
  ri = range(len(res))

  rv = 'res'
  # rv = 'train_res'

  df = pd.DataFrame({
    'p_plans': np.array([res[i][rv].p_plan for i in ri]),
    'earned_rew': np.array([res[i][rv].mean_total_reward for i in ri]),
    'reward_acc': np.array([res[i]['exploit_acc'] for i in ri]),
    'state_acc': np.array([res[i][rv].state_prediction_acc for i in ri]),
    'xs': np.array([res[i]['experience'] / 1e6 for i in ri]),
    'subdir': [res[i]['subdirectory'] for i in ri],
    'num_rollouts': [res[i]['num_entropy_rollouts'] for i in ri],
    'policy_entropies': [np.nanmean(res[i]['policy_entropies'], axis=0) for i in ri]
  })

  mask = df['subdir'] == 'plan-yes-full-60'
  subset = df.loc[mask, :]

  entropies = np.stack(subset['policy_entropies'].values)
  num_rollouts = subset['num_rollouts'].iloc[0]
  rollouts_str = [f'# Rollouts = {x}' for x in num_rollouts]

  analysis_multi(subset['xs'], entropies, rollouts_str, 'entropy', save_p)
  # analysis_scalar(subset['xs'], subset['p_plans'], 'p(plan)', save_p, ylim=[0., 0.6])
  # analysis_scalar(subset['xs'], subset['earned_rew'], 'mean reward', save_p, ylim=[0., 8.5])
  # analysis_scalar(subset['xs'], subset['reward_acc'], 'exploit reward pred acc', save_p)
  # analysis_scalar(subset['xs'], subset['state_acc'], 'state pred acc', save_p)