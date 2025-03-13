import eval
import env
from model import AgentModel
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time

def save_checkpoint(model: AgentModel, mazes, meta, ep_p: eval.EpisodeParams, save_p, fname):
  if not os.path.exists(save_p): os.makedirs(save_p, exist_ok=True)
  sd = {
    'state': model.state_dict(), 'params': model.ctor_params(), 
    'mazes': mazes, 'meta': meta, 'episode_params': ep_p
  }
  torch.save(sd, os.path.join(save_p, fname))

def b2s(b: bool): return 'yes' if b else 'no'

def main():
  """
  (begin) hps
  """
  root_dir = os.getcwd()
  save = True
  log = True
  prefer_gpu = False
  planning_enabled = True
  agent_chooses_ticks_enabled = False
  force_rollouts_at_start_of_exploit_phase = False
  ticks_take_time = False
  use_fixed_mazes = False
  s = 4 # arena side length
  batch_size = 40
  num_episodes = 50000 * 4
  hidden_size = 100
  recurrent_layer_type = 'gru'
  plan_len = 8  # long
  # plan_len = 4  # short
  rand_ticks = False
  num_ticks_per_step = 1
  planning_action_time = 0.12
  # num_ticks_per_step = 1
  lr = 1e-3
  """
  (end) hps
  """

  subdir = f'plan_{b2s(planning_enabled)}-hs_{hidden_size}-' + \
    f'plan_len_{plan_len}-rand_ticks_{b2s(rand_ticks)}-num_ticks_{num_ticks_per_step}-' + \
    f'recurrence_{recurrent_layer_type}-agent_chooses_ticks_{b2s(agent_chooses_ticks_enabled)}-' + \
    f'ticks_take_time_{b2s(ticks_take_time)}'
  
  device = torch.device('cuda:0' if prefer_gpu and torch.cuda.is_available() else 'cpu')

  if log: writer = SummaryWriter(os.path.join(root_dir, 'logs'))

  ep_p = eval.EpisodeParams(
    force_rollouts_at_start_of_exploit_phase=force_rollouts_at_start_of_exploit_phase,
    num_rollouts_per_planning_action=1,
    num_ticks_per_step_is_randomized=rand_ticks,
    num_ticks_per_step=num_ticks_per_step,
    verbose=2
  )

  meta = eval.make_meta(
    arena_len=s, plan_len=plan_len, device=device, 
    planning_enabled=planning_enabled, 
    agent_chooses_ticks_enabled=agent_chooses_ticks_enabled, 
    ticks_take_time=ticks_take_time, planning_action_time=planning_action_time)
  
  model = eval.build_model(
    meta=meta, hidden_size=hidden_size, recurrent_layer_type=recurrent_layer_type)

  fixed_mazes = env.build_fixed_maze_arenas(s, batch_size)

  optim = torch.optim.Adam(lr=lr, params=model.parameters())

  t0 = time.time()
  for e in range(num_episodes):
    tot_t = time.time() - t0
    print(f'{e+1} of {num_episodes} ({((e+1)/num_episodes*100):.3f}%); {tot_t:.3f}s')

    mazes = fixed_mazes if use_fixed_mazes else env.build_maze_arenas(s, batch_size)
    res = eval.run_episode(meta, model, mazes, params=ep_p)
    loss = res.loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    if log and e % int(25) == 0:
      writer.add_scalar('reward', res.mean_total_reward, e)
      writer.add_scalar('p(plan)', res.p_plan, e)

    if save and ((e == num_episodes - 1) or e % int(5e3) == 0):
      save_p = os.path.join(root_dir, 'checkpoints', subdir)
      save_checkpoint(model, mazes, meta, ep_p, save_p, f'cp-{e}.pth')

if __name__ == '__main__':
  main()