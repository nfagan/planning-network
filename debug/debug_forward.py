import sys; sys.path.append('..')
import model
import eval
import env
from utility import dataclass_to_dict
import torch
from dataclasses import asdict
from debug_utility import put_params
# import bson
from scipy.io import loadmat, savemat

def count_parameters(model: torch.nn.Module): 
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def arenas_from_kris_walls(kris_walls, s: int):
  fixed_mazes = []
  for bi in range(kris_walls.shape[2]):
    maze = env.Arena(s)
    for x in range(s):
      for y in range(s):
        si = env.coord_to_index(x, y, s)
        wc = kris_walls[si, :, bi]  
        maze.walls[y, x, :] = wc
    fixed_mazes.append(maze)
  return fixed_mazes

def main():
  kris_cp = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/N100_T50_Lplan8_seed62_1000-dst.mat'
  kris_walls_cp = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/walls.mat-dst.mat'
  kris_mdl_res_cp = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/output_N100_T50_Lplan8_seed62_1000-dst.mat'
  dst_p = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/my_output-dst.mat'

  s = 4 # arena side length
  # batch_size = 40

  ep_p = eval.EpisodeParams(
    force_rollouts_at_start_of_exploit_phase=False,
    num_rollouts_per_planning_action=1,
    num_ticks_per_step_is_randomized=False,
    num_ticks_per_step=1,
    sample_actions_greedily=True,
    verbose=2
  )

  meta = eval.make_meta(
    arena_len=s, plan_len=8, device=torch.device('cpu'), 
    planning_enabled=True, 
    agent_chooses_ticks_enabled=False, 
    include_dummy_prediction_output=True,
    ticks_take_time=False)
  
  model = eval.build_model(meta=meta, hidden_size=100, recurrent_layer_type='gru', use_zero_bias_hh=False)

  """
  """
  kris_cp = loadmat(kris_cp)
  kris_walls = loadmat(kris_walls_cp)['walls']
  mazes = arenas_from_kris_walls(kris_walls, s)
  kris_mdl_res = loadmat(kris_mdl_res_cp)
  put_params(model, kris_cp)
  """
  """

  x0 = torch.Tensor(kris_mdl_res['rx']).T
  h0 = torch.Tensor(kris_mdl_res['rh0']).T
  h1 = torch.tensor(kris_mdl_res['rh1']).T
  bs = h0.shape[0]
  nt = torch.ones((bs,), dtype=torch.long)
  my_outs = eval.forward_agent_model(
    meta=meta, model=model, x=x0, h_rnn=h0, num_ticks=nt, greedy_actions=True, disable_rollouts=False)
  mh1 = my_outs[0]
  abs_ds = (mh1 - h1).abs()
  max_delta = abs_ds.max()
  print(f'Max delta: {max_delta.item()}')
  # import pdb; pdb.set_trace()

  train_params = [*filter(lambda p: p.requires_grad, model.parameters())]
  optim = torch.optim.Adam(lr=1e-3, params=train_params)

  # mazes = env.build_fixed_maze_arenas(s, batch_size)
  res = eval.run_episode(meta, model, mazes, params=ep_p)
  loss = res.loss
  optim.zero_grad()
  loss.backward()

  resd = dataclass_to_dict(res)
  if False: savemat(dst_p, resd)

if __name__ == '__main__':
  main()