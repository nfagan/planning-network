from model import AgentModel
import eval
import env
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os

REPO_ROOT = '/Users/nick/Documents/mattarlab/project1/planning-network'

def load_model(cp_dir: str, cp_ind: int, meta: eval.Meta, hidden_size: int):
  # @TODO: serialize model attributes
  model = eval.build_model(meta=meta, hidden_size=hidden_size)
  cp_p = os.path.join(cp_dir, f'cp-{cp_ind}.pth')
  sd = torch.load(cp_p)
  model.load_state_dict(sd['state'])
  return model

# -------------------------------------------------------------------------------------

def analysis_scalar(
  meta: eval.Meta, models: List[AgentModel], mazes: List[env.Arena], xs: np.array, attr: str):
  # 
  s = []
  for i, model in enumerate(models):
    print(f'{i+1} of {len(models)}')
    res = eval.run_episode(meta, model, mazes, verbose=False)
    s.append(getattr(res, attr))

  plt.figure(1)
  plt.clf()
  plt.plot(xs, s)
  plt.show()

# -------------------------------------------------------------------------------------

def analysis_p_rollout(meta, models, mazes, xs):
  analysis_scalar(meta, models, mazes, xs, 'p_plan')

# -------------------------------------------------------------------------------------

def analysis_earned_reward(meta, models, mazes, xs):
  analysis_scalar(meta, models, mazes, xs, 'mean_total_reward')

# -------------------------------------------------------------------------------------

def analysis_state_pred_acc(meta, models, mazes, xs):
  analysis_scalar(meta, models, mazes, xs, 'state_prediction_acc')

# -------------------------------------------------------------------------------------

def analysis_reward_pred_acc(meta, models, mazes, xs):
  analysis_scalar(meta, models, mazes, xs, 'reward_prediction_acc')

# -------------------------------------------------------------------------------------

if __name__ == '__main__':
  dev = torch.device('cpu')
  batch_size = int(1e3)
  hidden_size = 100
  arena_len = 4
  meta = eval.make_meta(arena_len=arena_len, batch_size=batch_size, plan_len=8, device=dev)
  mazes = env.build_maze_arenas(arena_len, batch_size)

  cp_dir = os.path.join(REPO_ROOT, 'checkpoints')
  cp_inds = np.arange(0, int(7e4)+1, int(5e3))
  models = [load_model(cp_dir, i, meta, hidden_size) for i in cp_inds]

  tot_experience = cp_inds * 40 / 1e6 # @TODO: This batch size was fixed during training
  analysis_reward_pred_acc(meta, models, mazes, tot_experience)