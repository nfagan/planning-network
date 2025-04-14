import env
from env import move_agent
import torch
from utility import _is_deterministic
from typing import List

_NO_ACTION = -1
_DEBUG_START_STATE = 2
_DEBUG_REWARD_LOC = 8

class Environment(object):
  def __init__(self):
    pass

  def initialize(self): raise NotImplementedError
  def act(self, s: torch.Tensor, a1: torch.Tensor): raise NotImplementedError
  def get_obs(self): raise NotImplementedError
  def plan(self): raise NotImplementedError
  def batch_size(self): raise NotImplementedError
  def slice(self, idx: torch.Tensor): raise NotImplementedError

# ----------------------------------------------------------------------------------------

class MazeEnvironment(Environment):
  def __init__(self, *, mazes: List[env.Arena], num_states: int, planning_action: int, device):
    super().__init__()
    self.mazes = mazes
    self.num_states = num_states
    self.planning_action = planning_action
    self.device = device
    self.reward_location = None

  def initialize(self):
    return maze_initialize(self)

  def act(self, s: torch.Tensor, a1: torch.Tensor):
    return maze_act_concretely(self, s, a1)

  def batch_size(self):
    return len(self.mazes)
  
  def get_obs(self):
    kw = torch.stack([torch.tensor(x.walls) for x in self.mazes]).to(self.device)
    kw = kw.transpose(1, 2).flatten(1, 2)
    walls = kw[:, :, [0, 2]].transpose(1, 2).flatten(1)
    return walls
  
  def slice(self, idx: torch.Tensor) -> Environment:
    me = MazeEnvironment(
      mazes=[self.mazes[i] for i in idx],
      num_states=self.num_states,
      planning_action=self.planning_action,
      device=self.device
    )
    me.reward_location = self.reward_location[idx]
    return me
  
# ----------------------------------------------------------------------------------------

def maze_initialize(env: MazeEnvironment):
  batch_size = env.batch_size()
  a = torch.ones((batch_size,), dtype=torch.long).to(env.device) * _NO_ACTION
  rew_loc = torch.randint(0, env.num_states, (batch_size,)).to(env.device)
  s = torch.randint(0, env.num_states, (batch_size,)).to(env.device)
  if _is_deterministic(): rew_loc[:] = _DEBUG_REWARD_LOC; s[:] = _DEBUG_START_STATE
  # ensure agent does not begin at reward location
  s = maze_teleport_from_reward(s, env.num_states, rew_loc).to(env.device)
  env.reward_location = rew_loc
  return s, a

def maze_teleport_from_reward(s: torch.Tensor, n: int, rew_loc: torch.Tensor):
  i = s == rew_loc
  if _is_deterministic(): s[i] = _DEBUG_START_STATE; return s
  while torch.any(i):
    s[i] = torch.randint(0, n, (torch.sum(i),))
    i = s == rew_loc
  return s

def maze_act_concretely(env: MazeEnvironment, s: torch.Tensor, a1: torch.Tensor):
  """
  Move the agent in the environment, if its chosen action is not the planning action, and receive
  reward if the goal state is reached.
  """
  # @NOTE: See Kris' walls_build.jl/act_and_receive_reward
  rew_loc = env.reward_location
  batch_size = env.batch_size()
  at_rew = s == rew_loc
  s1 = torch.zeros_like(s)
  rew = torch.zeros((batch_size, 1))

  for i in range(batch_size):
    act = a1[i].item()
    if act == env.planning_action:
      s1[i] = s[i]
      continue
    # perform a concrete action (a movement)
    sn = move_agent(s[i].item(), act, env.mazes[i])
    if sn == rew_loc[i] and not at_rew[i]:  # newly reached the goal
      rew[i, 0] = 1.
    s1[i] = sn

  # @NOTE: s1p is the prediction target; s1 is the agent's true next state. the agent can be at the
  # goal location, in which case the prediction target ought to be the state it would have reached 
  # if it were "permitted" to act concretely.
  s1p = s1.clone()
  s1[at_rew] = maze_teleport_from_reward(s[at_rew], env.num_states, rew_loc[at_rew])
  return s1, s1p, rew, at_rew