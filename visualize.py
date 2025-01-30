from model import AgentModel
import eval
import env
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List

# 0: right | 1: left | 2: up | 3: down

def load_model(cp_dir: str, cp_ind: int) -> Tuple[AgentModel, eval.Meta, List[env.Arena]]:
  cp_p = os.path.join(cp_dir, f'cp-{cp_ind}.pth')
  sd = torch.load(cp_p)
  model = AgentModel.from_ctor_params(sd['params'])
  model.load_state_dict(sd['state'])
  mazes = sd['mazes']
  meta = sd['meta']
  return model, meta, mazes

def draw_arena(arena: env.Arena, goal: int):
  s = arena.s

  lp = {'linewidth': 1, 'color': 'black'}
  lpw = {'linewidth': 2, 'color': 'red'}

  for x in range(s):
    for y in range(s):
      i = env.coord_to_index(x, y, s)
      pr = lpw if not arena.traversible(i, 0) else lp
      pl = lpw if not arena.traversible(i, 1) else lp
      pu = lpw if not arena.traversible(i, 2) else lp
      pd = lpw if not arena.traversible(i, 3) else lp
      x0 = float(x) - 0.5
      x1 = x0 + 1.
      y0 = float(y) - 0.5
      y1 = y0 + 1.
      plt.plot([x1, x1], [y0, y1], **pr)
      plt.plot([x0, x0], [y0, y1], **pl)
      plt.plot([x0, x1], [y1, y1], **pu)
      plt.plot([x0, x1], [y0, y0], **pd)

  x, y = env.index_to_coord(goal, arena.s)
  plt.plot(x, y, 'g*')

def draw_step(arena: env.Arena, state: int, a: int, ms: float):
    s = 0.5
    x, y = env.index_to_coord(state, arena.s)
    plt.plot(x, y, 'ko', markersize=ms)
    if a == 0:
      plt.quiver(x, y, s, 0)
    elif a == 1:
      plt.quiver(x, y, -s, 0)
    elif a == 2:
      plt.quiver(x, y, 0, s)
    elif a == 3:
      plt.quiver(x, y, 0, -s)

def draw_trial(arena: env.Arena, states: torch.Tensor, actions: torch.Tensor, goal: int):
  ms = np.linspace(0, 1, len(states)) * 4. + 4.
  for i in range(len(states)):
    plt.clf()
    draw_arena(arena, goal)
    draw_step(arena, states[i].item(), actions[i].item(), ms[i])
    plt.show()

if __name__ == '__main__':
  subdir = 'plan-yes'
  cp_dir = os.path.join(os.getcwd(), 'checkpoints', subdir)

  model, meta, cp_mazes = load_model(cp_dir, int(95e3))
  new_mazes = env.build_maze_arenas(int(np.sqrt(meta.num_states)), 1)
  use_mazes = new_mazes
  arena = use_mazes[0]

  res = eval.run_episode(meta, model, use_mazes, verbose=0)

  active = res.actives[0, :].type(torch.bool)
  states = res.states0[0, active]
  actions = res.actions[0, active]

  plt.figure(1)
  draw_trial(arena, states, actions, res.reward_locs[0].item())