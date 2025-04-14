import sys; sys.path.append('..')
import eval
import env
from environment import MazeEnvironment
from evaluate import exploit_reward_state_prediction_accuracy_from_episode_result
from model import AgentModel
from debug_utility import put_params
from utility import dataclass_to_dict, filter_dict, split_array_indices
import torch
from scipy.io import loadmat, savemat
import numpy as np
import os
from itertools import product
from typing import List
import dataclasses
from dataclasses import dataclass
from multiprocessing import Process

NUM_PROCESSES = 5
# NUM_PROCESSES = 0

@dataclass
class Uniform:
  rep_dst_p = '/Users/nick/source/mattarlab/planning-network/results'
  cp_src_p = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models'
  do_save = True
  meta: eval.Meta
  ep_p: eval.EpisodeParams
  environ: MazeEnvironment
  hiddens: int

def compute_vf_error(rews: np.ndarray, vs: np.ndarray):
  cum_r = np.hstack((np.zeros((rews.shape[0], 1)), np.cumsum(rews[:, :-1], axis=1)))
  vt = np.sum(rews, axis=1, keepdims=True) - cum_r
  vf_err = np.abs(vs - vt)
  return vf_err

def eval_one(uniform: Uniform, model: AgentModel, fname: str):
  cp_p = os.path.join(uniform.cp_src_p, fname)
  cp = loadmat(cp_p)
  put_params(model, cp)

  res = eval.run_episode(uniform.meta, model, uniform.environ, params=uniform.ep_p)

  ep_p_no_rollouts = dataclasses.replace(uniform.ep_p, disable_rollouts=True)
  res_no_rollouts = eval.run_episode(uniform.meta, model, uniform.environ, params=ep_p_no_rollouts)

  variants = [res, res_no_rollouts]
  variant_info = [{'rollouts_enabled': True}, {'rollouts_enabled': False}]
  save_res = {'data': []}

  for i, res in enumerate(variants):
    resd = dataclass_to_dict(res)
    resd['vf_error'] = np.mean(compute_vf_error(resd['rewards'], resd['v']), axis=1)
    resd['exploit_reward_prediction_acc'] = exploit_reward_state_prediction_accuracy_from_episode_result(res)
    resd['variant_info'] = variant_info[i]
    keys = ['p_plan', 'mean_total_reward', 'vf_error',
            'state_prediction_acc', 'reward_prediction_acc', 'exploit_reward_prediction_acc',
            'variant_info']
    resd = filter_dict(resd, keys)
    save_res['data'].append(resd)
  
  if uniform.do_save:
    dst_p = os.path.join(uniform.rep_dst_p, f'replicate-{fname}')
    savemat(dst_p, save_res)

def eval_loop(uniform: Uniform, its: List):
  model = eval.build_model(
    meta=uniform.meta, hidden_size=uniform.hiddens, recurrent_layer_type='gru')
  model.eval()

  for i in range(its.shape[0]):
    print(f'{i+1} of {its.shape[0]}')
    seed, epoch = its[i, :]
    fname = f'N{uniform.hiddens}_T50_Lplan8_seed{seed}_{epoch}-dst.mat'
    eval_one(uniform, model, fname)

if __name__ == '__main__':
  hiddens = [60, 80, 100]
  seeds = [61, 62, 63, 64, 65]
  epochs = np.arange(11) * 100

  ep_p = eval.EpisodeParams(sample_actions_greedily=True)

  arena_len = 4
  num_episodes = 1_000
  meta = eval.make_meta(
    arena_len=arena_len, plan_len=8, device=torch.device('cpu'), 
    planning_enabled=True, 
    agent_chooses_ticks_enabled=False, 
    include_dummy_prediction_output=True,
    ticks_take_time=False)

  fixed_mazes = env.build_fixed_maze_arenas(arena_len, num_episodes)
  environ = MazeEnvironment(
    mazes=fixed_mazes, num_states=meta.num_states, 
    planning_action=meta.planning_action, device=meta.device)

  for hid in hiddens:
    its = np.array([*product(seeds, epochs)], dtype=object)
    un = Uniform(meta=meta, ep_p=ep_p, environ=environ, hiddens=hid)

    if NUM_PROCESSES <= 0:
      eval_loop(un, its)
    else:
      pi = split_array_indices(len(its), NUM_PROCESSES)
      process_args = [(un, its[x, :]) for x in pi]
      processes = [Process(target=eval_loop, args=args) for args in process_args]
      for p in processes: p.start()
      for p in processes: p.join()
