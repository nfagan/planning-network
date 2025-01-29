from model import AgentModel
import eval
import env
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

def load_checkpoint(cp_dir: str, cp_ind: int) -> Tuple[AgentModel, eval.Meta, List[env.Arena]]:
  cp_p = os.path.join(cp_dir, f'cp-{cp_ind}.pth')
  sd = torch.load(cp_p)
  model = AgentModel.from_ctor_params(sd['params'])
  model.load_state_dict(sd['state'])
  if 'meta' in sd:
    meta = sd['meta']
  else: # @TODO: remove this
    meta = eval.make_meta(arena_len=4, plan_len=8, device=None, planning_enabled=False)
  return model, meta, sd['mazes']

def find_rewards(rews: torch.Tensor):
  trials = []
  for i in range(rews.shape[0]):
    ri = torch.argwhere(rews[i, :]).squeeze(1)
    if ri.numel() > 0:
      trials.append([i.item() for i in ri])
    else:
      trials.append([])
  return trials

def find_first_exploit(rews: List[List[int]]):
  exploit_ind = np.ones((len(rews),), dtype=int) * -1
  for i, t in enumerate(rews):
    if len(t) > 1:
      exploit_ind[i] = t[1]
  return exploit_ind

def exploit_reward_state_prediction_accuracy(
  exploit_ind: np.ndarray, rew_locs: torch.Tensor, pred_rew: torch.Tensor):
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
  mazes = env.build_maze_arenas(arena_len, batch_size)

  subdir = 'plan-yes'
  cp_dir = os.path.join(os.getcwd(), 'checkpoints', subdir)
  dst_dir = os.path.join(os.getcwd(), 'results')

  cp_inds = np.arange(0, int(95e3)+1, int(5e3))
  tot_experience = cp_inds * 40 # @TODO: This batch size was fixed during training

  rows = []
  for i in range(len(cp_inds)):
    print(f'{i+1} of {len(cp_inds)}')
    model, meta, cp_mazes = load_checkpoint(cp_dir, cp_inds[i])
    res = eval.run_episode(meta, model, mazes, verbose=0)
    train_res = eval.run_episode(meta, model, cp_mazes, verbose=0)
    ri = find_rewards(res.rewards)
    first_exploit = find_first_exploit(ri)
    exploit_acc = exploit_reward_state_prediction_accuracy(
      first_exploit, res.reward_locs, res.predicted_rewards)
    row = {
      'res': res,
      'train_res': train_res,
      'first_exploit': first_exploit,
      'exploit_acc': exploit_acc,
      'experience': tot_experience[i]
    }
    rows.append(row)
  torch.save({'rows': rows}, os.path.join(dst_dir, 'evaluation.pth'))