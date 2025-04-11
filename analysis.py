import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Context:
  save_p: Optional[str]
  show_plot: bool

def load_results(root_p):
  eval_p = os.path.join(root_p, 'results')
  eval_fs = glob.glob(f'{eval_p}/*.pth')
  res = [torch.load(x)['row'] for x in eval_fs]
  res = sorted(res, key=lambda x: x['experience'])
  return res

def analysis_scalar(xs, ys, ylab, context: Context, ylim=None):
  f = plt.figure(1)
  plt.clf()
  plt.plot(xs, ys)
  plt.ylabel(ylab)
  if ylim is not None: plt.ylim(ylim)
  if context.show_plot: plt.show()
  plt.draw()
  if context.save_p is not None:
    f.savefig(os.path.join(context.save_p, f'{ylab}.png'))

def analysis_multi(xs, ys, by, ylab, context: Context, ylim=None):
  f = plt.figure(1)
  plt.clf()
  h = plt.plot(xs, ys)
  plt.ylabel(ylab)
  num_lines = len(by)
  cmap = plt.get_cmap('spring', num_lines)(np.arange(num_lines))[:, :3]
  for i in range(num_lines):
    h[i].set_color(cmap[i, :])
    h[i].set_label(f'{by[i]}')
  if ylim is not None: plt.ylim(ylim)
  plt.legend()
  if context.show_plot: plt.show()
  plt.draw()
  if context.save_p is not None:
    f.savefig(os.path.join(context.save_p, f'{ylab}.png'))

def to_data_frame(res):
  """
  Convert results across checkpoints to a dataframe
  """
  ri = range(len(res))

  rv = 'res'
  # rv = 'train_res'

  def guard_none_mean(key):
    return [np.nanmean(res[i][key], axis=0) if res[i][key] is not None else np.zeros((1,)) for i in ri]

  df = pd.DataFrame({
    'p_plans': np.array([res[i][rv].p_plan for i in ri]),
    'earned_rew': np.array([res[i][rv].mean_total_reward for i in ri]),
    'reward_acc': np.array([res[i]['exploit_acc'] for i in ri]),
    'state_acc': np.array([res[i][rv].state_prediction_acc for i in ri]),
    'xs': np.array([res[i]['experience'] / 1e6 for i in ri]),
    'subdir': [res[i]['subdirectory'] for i in ri],
    'num_rollouts': [res[i]['num_forced_rollouts'] for i in ri],
    'policy_entropies': [np.nanmean(res[i]['forced_rollout_policy_entropies'], axis=0) for i in ri],
    'num_ticks': [res[i]['num_ticks'] if res[i]['num_ticks'] is not None else np.zeros((1,)) for i in ri],
    'exploit_only_forced_ticks_mean_reward': guard_none_mean('exploit_only_forced_ticks_mean_reward'),
    'explore_only_forced_ticks_mean_reward': guard_none_mean('explore_only_forced_ticks_mean_reward'),
    'always_forced_ticks_mean_reward': guard_none_mean('always_forced_ticks_mean_reward'),
    'once_randomly_forced_ticks_mean_reward': guard_none_mean('once_randomly_forced_ticks_mean_reward')
  })

  return df

def main():
  root_p = os.getcwd()

  cp_subdir_names = [
    'plan_yes-hs_100-plan_len_8-rand_ticks_no-num_ticks_1-recurrence_gru-agent_chooses_ticks_no-ticks_take_time_no-20250409-102003'
  ]

  for si, cp_subdir_name in enumerate(cp_subdir_names):
    print(f'{cp_subdir_name} ({si+1} of {len(cp_subdir_names)})')

    save_p = os.path.join(root_p, 'plots', cp_subdir_name)
    if save_p is not None: os.makedirs(save_p, exist_ok=True)

    ctx = Context(save_p=save_p, show_plot=False)

    res = load_results(root_p)
    df = to_data_frame(res)

    mask = df['subdir'] == cp_subdir_name
    subset = df.loc[mask, :]

    entropies = np.stack(subset['policy_entropies'].values)
    num_rollouts = subset['num_rollouts'].iloc[0]
    rollouts_str = [f'# Rollouts = {x}' for x in num_rollouts]

    exploit_only_mean_rews = np.stack(subset['exploit_only_forced_ticks_mean_reward'].values)
    explore_only_mean_rews = np.stack(subset['explore_only_forced_ticks_mean_reward'].values)
    random_once_mean_rews = np.stack(subset['once_randomly_forced_ticks_mean_reward'].values)
    forced_ticks_mean_rews = np.stack(subset['always_forced_ticks_mean_reward'].values)
    
    num_ticks = subset['num_ticks'].iloc[0]
    ticks_str = [f'# ticks = {x}' for x in num_ticks]
    rew_y = [0., 10.]
    xs = subset['xs']

    analysis_multi(xs, entropies, rollouts_str, 'entropy', ctx)
    analysis_multi(xs, random_once_mean_rews, ticks_str, 'ticks mean reward (randomly once)', ctx, ylim=rew_y)
    analysis_multi(xs, exploit_only_mean_rews, ticks_str, 'ticks mean reward (exploit only)', ctx, ylim=rew_y)
    analysis_multi(xs, explore_only_mean_rews, ticks_str, 'ticks mean reward (explore only)', ctx, ylim=rew_y)
    analysis_multi(xs, forced_ticks_mean_rews, ticks_str, 'ticks mean reward', ctx, ylim=rew_y)
    analysis_scalar(xs, subset['p_plans'], 'p(plan)', ctx, ylim=[0., 0.6])
    analysis_scalar(xs, subset['earned_rew'], 'mean reward', ctx, ylim=[0., 8.5])
    analysis_scalar(xs, subset['reward_acc'], 'exploit reward pred acc', ctx, ylim=[0, 1])
    analysis_scalar(xs, subset['state_acc'], 'state pred acc', ctx, ylim=[0, 1])

if __name__ == '__main__':
  main()