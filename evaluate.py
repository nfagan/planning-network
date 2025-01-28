from model import AgentModel
import eval
import env
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os

REPO_ROOT = '/Users/nick/Documents/mattarlab/project1/planning-network'

def load_model(cp_dir: str, cp_ind: int):
  cp_p = os.path.join(cp_dir, f'cp-{cp_ind}.pth')
  sd = torch.load(cp_p)
  model = AgentModel.from_ctor_params(sd['params'])
  model.load_state_dict(sd['state'])
  return model

# -------------------------------------------------------------------------------------

def find_rewards(rews: torch.Tensor):
  trials = []
  for i in range(rews.shape[0]):
    ri = torch.argwhere(rews[i, :]).squeeze(1)
    if ri.numel() > 0:
      trials.append([i.item() for i in ri])
    else:
      trials.append([])
  return trials

# -------------------------------------------------------------------------------------

def find_first_exploit(rews: List[List[int]]):
  exploit_ind = np.ones((len(rews),), dtype=int) * -1
  for i, t in enumerate(rews):
    if len(t) > 1:
      exploit_ind[i] = t[1]
  return exploit_ind

# -------------------------------------------------------------------------------------

def exploit_reward_state_prediction_accuracy(
  exploit_ind: np.array, rew_locs: torch.Tensor, pred_rew: torch.Tensor):
  #
  n = 0
  d = 0
  for i in range(exploit_ind.shape[0]):
    ei = exploit_ind[i]
    if ei < 0: continue
    rew = rew_locs[i]
    pred = pred_rew[i, ei:]
    n += torch.sum(rew == pred).item()
    d += pred.numel()
  return 0. if d == 0 else n/d

# -------------------------------------------------------------------------------------

if __name__ == '__main__':
  dev = torch.device('cpu')
  batch_size = int(1e3)
  arena_len = 4
  meta = eval.make_meta(arena_len=arena_len, batch_size=batch_size, plan_len=8, device=dev)
  mazes = env.build_maze_arenas(arena_len, batch_size)

  cp_dir = os.path.join(REPO_ROOT, 'checkpoints/plan-yes2')
  dst_dir = os.path.join(REPO_ROOT, 'results')

  cp_inds = np.arange(0, int(7e4)+1, int(5e3))
  tot_experience = cp_inds * 40 # @TODO: This batch size was fixed during training
  models = [load_model(cp_dir, i) for i in cp_inds]

  rows = []
  for i, model in enumerate(models):
    print(f'{i+1} of {len(models)}')
    res = eval.run_episode(meta, model, mazes, verbose=False)
    ri = find_rewards(res.rewards)
    first_exploit = find_first_exploit(ri)
    exploit_acc = exploit_reward_state_prediction_accuracy(
      first_exploit, res.reward_locs, res.predicted_rewards)
    row = {
      'res': res,
      'first_exploit': first_exploit,
      'exploit_acc': exploit_acc,
      'experience': tot_experience[i]
    }
    rows.append(row)
  torch.save({'rows': rows}, os.path.join(dst_dir, 'evaluation.pth'))