import sys; sys.path.append('..')
from debug_utility import put_params, arenas_from_kris_walls
import eval
import env
from utility import dataclass_to_dict, _set_deterministic
from environment import MazeEnvironment
import torch
from scipy.io import loadmat, savemat
import numpy as np
import builtins
from typing import List

def create_maze_environment(meta: eval.Meta, mazes: List[env.Arena]):
  return MazeEnvironment(
    mazes=mazes, num_states=meta.num_states, 
    planning_action=meta.planning_action, device=meta.device)

def grad_keys():
  def keys():
    # '_gru_core.bias_n': 'gradient:value:4:gs_k',
    return {
      'rnn.gru.weight_ih_l0': 'gradient:value:1:gs_k',
      'rnn.gru.weight_hh_l0': 'gradient:value:2:gs_k',
      'rnn.gru.bias_ih_l0': 'gradient:value:3:gs_k',
      'rnn.gru.bias_hh_l0': 'gradient:value:4:gs_k',

      'policy.linear.weight': 'gradient:value:5:gs_k',
      'policy.linear.bias': 'gradient:value:6:gs_k',

      'prediction.linear0.weight': 'gradient:value:7:gs_k',
      'prediction.linear0.bias': 'gradient:value:8:gs_k',
      'prediction.linear1.weight': 'gradient:value:9:gs_k',
      'prediction.linear1.bias': 'gradient:value:10:gs_k',
    }
  grad_key = keys()
  for k in grad_key.keys():
    v = grad_key[k].replace(':', '_')
    grad_key[k] = v
  return grad_key

def get_grads(mdl):
  p = grad_keys()
  res = {}
  for k in p.keys():
    ks = f"locals()['param'] = mdl.{k}; res[k] = param.grad"
    builtins.exec(ks)
  return res

def main():
  _set_deterministic(True)

  kris_cp = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/gradients_batch_1-dst.mat'
  kris_walls_cp = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/train-walls-dst.mat'

  s = 4 # arena side length

  ep_p = eval.EpisodeParams(sample_actions_greedily=True, verbose=2)

  meta = eval.make_meta(
    arena_len=s, plan_len=8, device=torch.device('cpu'), 
    planning_enabled=True, 
    agent_chooses_ticks_enabled=False, 
    include_dummy_prediction_output=True,
    ticks_take_time=False)
  
  model = eval.build_model(
    meta=meta, hidden_size=100, recurrent_layer_type='gru', use_zero_bias_hh=True)

  kris_cp = loadmat(kris_cp)
  put_params(model, kris_cp)

  kris_walls = loadmat(kris_walls_cp)['walls']
  mazes = arenas_from_kris_walls(kris_walls, s)
  environ = create_maze_environment(meta, mazes)

  train_params = [*filter(lambda p: p.requires_grad, model.parameters())]
  print('Num trainable params: ', len(train_params))

  optim = torch.optim.Adam(lr=1e-3, params=train_params)

  res = eval.run_episode(meta, model, environ, params=ep_p)
  loss = res.loss
  optim.zero_grad()
  loss.backward()

  grads = get_grads(model)
  gradk = grad_keys()
  for k in gradk.keys():
    key = gradk[k]
    kgrad = kris_cp[key]
    if kgrad.shape[1] == 1: kgrad = kgrad.squeeze(1)
    ngrad = grads[k]
    if ngrad is None: 
      print(f'My model lacks gradient for: {k}')
      continue

    ngrad = ngrad.clone().detach().cpu().numpy()
    if kgrad.shape == ngrad.shape:
      gdiff = np.abs(kgrad - ngrad)
      maxdiff = gdiff.max()
      print(f'Max gradient diff for: {k}: {maxdiff}')
    else:
      print(f'Shape mismatch for {k}: my shape: {ngrad.shape}; kris shape: {kgrad.shape}')
      if k != 'rnn.gru.bias_hh_l0':
        assert False, 'Grad shapes mismatch'

  resd = dataclass_to_dict(res)
  if False: savemat(dst_p, resd)

if __name__ == '__main__':
  main()