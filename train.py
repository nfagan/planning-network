from model import AgentModel, RNN, Policy, Prediction
import env
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
"""
Use batch as first dimension.

* act in environment
  . planning algorithm
* training
  . losses
"""

@dataclass
class Meta:
  num_actions: int
  batch_size: int
  num_inputs: int
  num_outputs: int
  num_states: int
  plan_len: int

def model_tree_search(*, meta: Meta, model: AgentModel, h0: torch.tensor):
  for _ in range(meta.plan_len):
    h, log_pi, v, pred_output, a = forward_agent_model(meta=meta, model=model)

def forward_agent_model(*, meta: Meta, model: AgentModel, x: torch.tensor, h0: torch.tensor):
  """
  a2c.jl/forward_modular
  h_rnn, ytemp = mod.network[GRUind].cell(h_rnn, x) #forward pass through recurrent part of RNN
  logπ_V = mod.policy(ytemp) #apply policy (and value) network
  logπ = logπ_V[1:Naction, :] #policy is the first few rows
  V = logπ_V[(Naction+1):(Naction+1), :] #value function is the next row
  ...
  prediction_input = [ytemp; ahot] #input to prediction module (concatenation of hidden state and action)
  prediction_output = mod.prediction(prediction_input) #output of prediction module
  return h_rnn, [logπ; V; prediction_output], a # return hidden state, network output, and sampled action
  """
  def _policy_subset(log_pi_v): return log_pi_v[:, :5]
  def _v_subset(log_pi_v): return log_pi_v[:, 5]
  def sample_actions(ps): return torch.multinomial(ps, 1, True).squeeze()

  h_rnn, ytemp = model.rnn(x, h0)
  log_pi_v = model.policy(ytemp)
  log_pi = _policy_subset(log_pi_v)
  v = _v_subset(log_pi_v)
  ps = torch.softmax(log_pi, dim=1)
  a = sample_actions(ps)
  ah = F.one_hot(a, meta.num_actions)
  pred_input = torch.hstack([ytemp, ah])
  pred_output = model.prediction(pred_input)
  return h_rnn, log_pi, v, pred_output, a

def gen_input(
    *, meta: Meta, arenas: List[env.Arena], prev_ahot: torch.tensor, 
    prev_rewards: torch.tensor, time: torch.tensor, shot: torch.tensor):
  """
  walls.jl/gen_input:
  ### speed this up ###
  x = zeros(Nin, batch)
  x[1:Naction, :] = ahot
  x[Naction + 1, :] = rew[:]
  x[Naction + 2, :] = world_state.environment_state.time / 50f0 #smaller time input in [0,1]
  x[(Naction + 3):(Naction + 2 + Nstates), :] = shot
  x[(Naction + 2 + Nstates + 1):(Naction + 2 + Nstates + Nwall_in), :] = wall_input

  if length(plan_input) > 0 #set planning input
      x[(Naction + 2 + Nstates + Nwall_in + 1):(Naction + 2 + Nstates + Nwall_in + Nplan_in), :] = 
        world_state.planning_state.plan_input
  end

  function get_wall_input(state, wall_loc)
    #state is 2xB
    #wall_loc is Nstates x 4 x B (4 is right/left/up/down)
    input = [wall_loc[:, 1, :]; wall_loc[:, 3, :]] #all horizontal and all vertical walls
    return input # don't take gradients of this
  end
  """
  walls = torch.vstack([torch.tensor(x.walls[:, :, [0, 2]].flatten()) for x in arenas])
  x = torch.zeros((meta.batch_size, meta.num_inputs))
  x[:, :meta.num_actions] = prev_ahot
  x[:, meta.num_actions:meta.num_actions+1] = prev_rewards
  x[:, meta.num_actions+1:meta.num_actions+2] = time
  x[:, meta.num_actions+2:meta.num_actions+2+meta.num_states] = shot
  x[:, meta.num_actions+2+meta.num_states:meta.num_actions+2+meta.num_states+meta.num_states*2] = walls
  # @TODO: Planning inputs
  return x

def meta_sizes(*, arena_len: int, batch_size: int, plan_len: int) -> Meta:
  """
  walls_build.jl/useful_dimensions:
  Nstates = Larena^2 #number of states in arena
  Nstate_rep = 2 #dimensionality of the state representation (e.g. '2' for x,y-coordinates)
  Naction = 5 #number of actions available
  Nout = Naction + 1 + Nstates #actions and value function and prediction of state
  Nout += 1 # needed for backward compatibility (this lives between state and reward predictions)
  Nwall_in = 2 * Nstates #provide full info
  Nin = Naction + 1 + 1 + Nstates + Nwall_in #5 actions, 1 rew, 1 time, L^2 states, some walls

  --- planning.jl/build_planner
  Nplan_in = 4*Lplan+1 #action sequence and whether we ended at the reward location
  Nplan_out = Nstates #rew location
  ---

  Nin += planner.Nplan_in #additional inputs from planning
  Nout += planner.Nplan_out #additional outputs for planning
  """
  num_actions = 5
  num_states = arena_len * arena_len
  num_walls = 2 * num_states
  num_inputs = num_actions + 1 + 1 + num_states + num_walls # + 1 rew, + 1 time
  num_outputs = num_actions + 1 + num_states

  num_inputs += 4 * plan_len + 1 #action sequence and whether we ended at the reward location
  num_outputs += num_states #(imagined) rew location

  return Meta(
    num_inputs=num_inputs, num_outputs=num_outputs, 
    num_actions=num_actions, num_states=num_states, batch_size=batch_size)

def build_model(*, meta: Meta) -> AgentModel:
  hs = 100
  num_inputs, num_outputs, num_actions = meta.num_inputs, meta.num_outputs, meta.num_actions
  policy_output_size = num_actions + 1  # +1 for value
  # i.e., only predict state occupancy (ignore policy + value outputs)
  pred_output_size = num_outputs - num_actions - 1 

  rnn = RNN(in_size=num_inputs, hidden_size=hs)
  policy = Policy(rnn_hidden_size=hs, output_size=policy_output_size)
  prediction = Prediction(input_size=hs + num_actions, output_size=pred_output_size)
  return AgentModel(rnn=rnn, policy=policy, prediction=prediction)

def main():
  s = 4
  batch_size = 128
  device = torch.device('cpu')
  mazes = [env.build_maze_arena(s) for _ in range(batch_size)]
  meta = meta_sizes(arena_len=s, batch_size=batch_size, plan_len=8)
  model = build_model(meta=meta)
  prev_ahot = torch.zeros((meta.batch_size, meta.num_actions))
  prev_rewards = torch.zeros((meta.batch_size, 1))
  time = torch.zeros((meta.batch_size, 1))
  s0 = torch.randint(0, meta.num_states, (meta.batch_size,))
  shot = F.one_hot(s0, meta.num_states)
  x = gen_input(
    meta=meta, arenas=mazes, prev_ahot=prev_ahot, 
    prev_rewards=prev_rewards, shot=shot, time=time)
  h0 = model.rnn.make_h0(meta.batch_size, device)
  h_rnn, log_pi, v, pred_output, a1 = forward_agent_model(meta=meta, model=model, x=x, h0=h0)
  # 
  s1 = torch.zeros_like(s0).detach()
  for i in range(batch_size):
    a = a1[i].item()
    if a < 4: 
      s1[i] = env.move_agent(s0[i].item(), a, mazes[i])
    else:
      # do planning
      pass
  import pdb; pdb.set_trace()

if __name__ == '__main__':
  main()