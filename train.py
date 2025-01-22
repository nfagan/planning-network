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
  num_concrete_actions: int # excludes planning action
  batch_size: int
  num_inputs: int
  num_planning_inputs: int
  num_planning_outputs: int
  num_outputs: int
  num_states: int
  plan_len: int
  planning_action = 4
  device: torch.device

def sample_actions(ps: torch.Tensor): return torch.multinomial(ps, 1, True).squeeze()

def make_meta(*, arena_len: int, batch_size: int, plan_len: int, device: torch.device) -> Meta:
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
  num_concrete_actions = 4  # movements; excludes planning action
  num_states = arena_len * arena_len
  num_walls = 2 * num_states
  num_inputs = num_actions + 1 + 1 + num_states + num_walls # + 1 rew, + 1 time
  num_outputs = num_actions + 1 + num_states

  # (one-hot) action sequence and whether we ended at the reward location
  num_planning_inputs = 4 * plan_len + 1
  # (imagined) reward location
  num_planning_outputs = num_states 

  num_inputs += num_planning_inputs
  num_outputs += num_planning_outputs

  return Meta(
    num_inputs=num_inputs, num_outputs=num_outputs, 
    num_actions=num_actions, num_states=num_states, 
    batch_size=batch_size, plan_len=plan_len, 
    num_concrete_actions=num_concrete_actions, 
    num_planning_inputs=num_planning_inputs, num_planning_outputs=num_planning_outputs,
    device=device)

def forward_agent_model(*, meta: Meta, model: AgentModel, x: torch.Tensor, h0: torch.Tensor):
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
  def _policy_subset(log_pi_v): return log_pi_v[:, :meta.num_actions]
  def _v_subset(log_pi_v): return log_pi_v[:, meta.num_actions]

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

def gen_plan_input(*, meta: Meta, path_hot: torch.Tensor):
  x = torch.zeros((path_hot.shape[0], meta.num_planning_inputs))
  # @TODO: found reward input should be after the path
  x[:, :meta.num_planning_inputs-1] = path_hot
  return x

def gen_input(
    *, meta: Meta, arenas: List[env.Arena], prev_ahot: torch.Tensor, 
    prev_rewards: torch.Tensor, time: torch.Tensor, shot: torch.Tensor, 
    plan_input: torch.Tensor = None):
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
  x = torch.zeros((len(arenas), meta.num_inputs))
  x[:, :meta.num_actions] = prev_ahot
  x[:, meta.num_actions:meta.num_actions+1] = prev_rewards
  x[:, meta.num_actions+1:meta.num_actions+2] = time
  x[:, meta.num_actions+2:meta.num_actions+2+meta.num_states] = shot
  x[:, meta.num_actions+2+meta.num_states:meta.num_actions+2+meta.num_states+meta.num_states*2] = walls
  if plan_input is not None:
    x[:, meta.num_actions+2+meta.num_states+meta.num_states*2:] = plan_input
  return x

def model_tree_search(
    *, meta: Meta, model: AgentModel, arenas: List[env.Arena], 
    h_rnn: torch.Tensor, s: torch.Tensor, a: torch.Tensor, 
    rewards: torch.Tensor, time: torch.Tensor):
  # 
  batch_size = len(arenas)
  path = torch.zeros((batch_size, meta.plan_len), dtype=torch.long)

  # for the first planning iteration, the policy derives from the rnn's current hidden state;
  # afterwards, the policy derives from the rnn's output.
  ytemp = h_rnn
  for pi in range(meta.plan_len):
    if pi > 0:
      shot = F.one_hot(s, meta.num_states)
      ahot = F.one_hot(a, meta.num_actions)
      x = gen_input(
        meta=meta, arenas=arenas, prev_ahot=ahot, 
        prev_rewards=rewards, time=time, shot=shot)
      h_rnn, ytemp = model.rnn(x, h_rnn)

    log_pi_v = model.policy(ytemp)
    # ignore thinking action
    log_pi = log_pi_v[:, :meta.num_concrete_actions]
    # sample next actions
    ps = torch.softmax(log_pi, dim=1)
    a = sample_actions(ps)
    ah = F.one_hot(a, meta.num_actions)

    # record chosen actions
    for b in range(batch_size):
      path[b, pi] = a[b]

    # predict and sample new states
    pred_input = torch.hstack([ytemp, ah])
    pred_output = model.prediction(pred_input)
    sn = torch.softmax(pred_output[:, :meta.num_states], dim=1)
    s = torch.argmax(sn, dim=1)
    time = time + 1.

  return path

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

def run_episode(meta: Meta, model: AgentModel, mazes: List[env.Arena]):
  a = torch.randint(0, meta.num_actions, (meta.batch_size,))
  s = torch.randint(0, meta.num_states, (meta.batch_size,))
  prev_rewards = torch.zeros((meta.batch_size, 1))
  time = torch.zeros((meta.batch_size, 1))
  h_rnn = model.rnn.make_h0(meta.batch_size, meta.device)

  loss = 0.0
  for t in range(20):
    plan_input = None
    if t > 0: plan_input = gen_plan_input(meta=meta, path_hot=paths_hot)

    prev_ahot = F.one_hot(a, meta.num_actions)
    shot = F.one_hot(s, meta.num_states)

    # perform a step of recurrent processing
    x = gen_input(
      meta=meta, arenas=mazes, prev_ahot=prev_ahot, 
      prev_rewards=prev_rewards, shot=shot, time=time, plan_input=plan_input)
    h_rnn, log_pi, v, pred_output, a1 = forward_agent_model(meta=meta, model=model, x=x, h0=h_rnn)
    # pred_state, pred_reward = from_prediction_output(pred_output)

    # update agent states
    s1 = torch.zeros_like(s).detach()
    for i in range(meta.batch_size):
      act = a1[i].item()
      if act < meta.planning_action: s1[i] = env.move_agent(s[i].item(), act, mazes[i])

    # plan
    pi = torch.argwhere(a1 == meta.planning_action).squeeze() # indices of planning actions
    path = model_tree_search(
      meta=meta, model=model, arenas=[mazes[i] for i in pi], 
      h_rnn=h_rnn[pi, :], s=s[pi], a=a[pi], rewards=prev_rewards[pi, :], time=time[pi, :])
    paths_hot = torch.zeros((meta.batch_size, path.shape[1] * meta.num_concrete_actions))
    paths_hot[pi, :] = F.one_hot(path, meta.num_concrete_actions).flatten(1).type(paths_hot.dtype)
    
    # prepare next iteration
    a = a1
    s = s1
    # prev_rewards = ...
    time = time + 1.

    # update losses
  import pdb; pdb.set_trace()

def main():
  s = 4 # arena side length
  batch_size = 128
  device = torch.device('cpu')

  meta = make_meta(arena_len=s, batch_size=batch_size, plan_len=8, device=device)
  model = build_model(meta=meta)
  mazes = [env.build_maze_arena(s) for _ in range(meta.batch_size)]
  
  run_episode(meta, model, mazes)

if __name__ == '__main__':
  main()