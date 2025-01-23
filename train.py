from model import AgentModel, RNN, Policy, Prediction
import env
import numpy as np
import torch
import torch.optim
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

def sample_actions(ps: torch.Tensor): 
  r = torch.multinomial(ps, 1, True)
  return r.squeeze(1)

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

def rollout(
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

def decompose_prediction_output(meta: Meta, pred_output: torch.Tensor):
  pred_s = pred_output[:, :meta.num_states]
  pred_r = pred_output[:, meta.num_states:]
  return pred_s, pred_r

def state_prediction_loss(true_s: torch.Tensor, pred_s: torch.Tensor):
  logits = pred_s.permute(0, 2, 1).flatten(0, 1)
  targets = true_s.flatten()
  return F.cross_entropy(logits, targets, reduction='sum')

def reward_prediction_loss(true_rew_s: torch.Tensor, pred_r: torch.Tensor):
  logits = pred_r.permute(0, 2, 1).flatten(0, 1)
  targets = true_rew_s.flatten()
  return F.cross_entropy(logits, targets, reduction='sum')

def td_error(rews: torch.Tensor, v: torch.Tensor):
  N = rews.shape[1]
  r = torch.zeros((rews.shape[0],), device=v.device)
  ds = torch.zeros_like(v)
  for t in range(N):
    r = rews[:, N - t - 1] + r
    ds[:, N - t - 1] = r - v[:, N - t - 1]
  return ds

def run_episode(meta: Meta, model: AgentModel, mazes: List[env.Arena]):
  """
  "Additionally, the time within the session was updated by only 120 ms after a rollout in contrast 
  to the 400-ms update after a physical action or teleportation step"
  """
  a = torch.randint(0, meta.num_actions, (meta.batch_size,)).to(meta.device)
  s = torch.randint(0, meta.num_states, (meta.batch_size,)).to(meta.device)
  # @TODO: Make sure s != rew_loc to start
  rew_loc = torch.randint(0, meta.num_states, (meta.batch_size,)).to(meta.device)

  prev_rewards = torch.zeros((meta.batch_size, 1)).to(meta.device)
  time = torch.zeros((meta.batch_size, 1)).to(meta.device)
  h_rnn = model.rnn.make_h0(meta.batch_size, meta.device)

  actions = []
  s0s = []  # states at step t
  s1s = []  # states at step t+1
  log_pis = []
  rews = []
  vs = []
  pred_states = []
  pred_rewards = []

  T = 20.0 # @TODO
  concrete_action_time = 0.4    # @TODO
  planning_action_time = 0.12   # @TODO

  t = 0
  p_plan = d_plan = 0.0
  while torch.any(time < T):
    plan_input = None
    if t > 0: plan_input = gen_plan_input(meta=meta, path_hot=paths_hot)

    prev_ahot = F.one_hot(a, meta.num_actions)
    shot = F.one_hot(s, meta.num_states)

    # perform a step of recurrent processing
    x = gen_input(
      meta=meta, arenas=mazes, prev_ahot=prev_ahot, 
      prev_rewards=prev_rewards, shot=shot, time=time, plan_input=plan_input)
    h_rnn, log_pi, v, pred_output, a1 = forward_agent_model(meta=meta, model=model, x=x, h0=h_rnn)
    pred_state, pred_reward = decompose_prediction_output(meta, pred_output)

    # update agent states
    s1 = torch.zeros_like(s)
    rew = torch.zeros_like(prev_rewards)

    for i in range(meta.batch_size):
      act = a1[i].item()
      if act == meta.planning_action: continue
      # perform a concrete action (a movement)
      sn = env.move_agent(s[i].item(), act, mazes[i])
      if sn == rew_loc[i]:  # reached the goal
        # teleport the agent to a *non-goal state* (@TODO)
        sn = torch.randint(0, meta.num_states, (1,))
        # update reward
        rew[i, 0] = 1.
      s1[i] = sn

    # plan
    paths_hot = torch.zeros((meta.batch_size, meta.plan_len * meta.num_concrete_actions))
    pi = torch.argwhere(a1 == meta.planning_action).squeeze(1) # indices of planning actions
    if pi.numel() > 0:
      with torch.no_grad():
        path = rollout(
          meta=meta, model=model, arenas=[mazes[i] for i in pi], s=s[pi], a=a[pi], 
          h_rnn=h_rnn[pi, :], rewards=prev_rewards[pi, :], time=time[pi, :])
      paths_hot[pi, :] = F.one_hot(path, meta.num_concrete_actions).flatten(1).type(paths_hot.dtype)
    p_plan += pi.numel()
    d_plan += meta.batch_size # @TODO: + num_active

    # increment time
    dt = torch.ones_like(time) * concrete_action_time
    dt[pi, :] = planning_action_time

    # push results
    rews.append(rew)
    log_pis.append(log_pi)
    vs.append(v)
    actions.append(a)
    s0s.append(s)
    s1s.append(s1)
    pred_states.append(pred_state)
    pred_rewards.append(pred_reward)
    
    # prepare next iteration
    a = a1
    s = s1
    prev_rewards = rew
    time = time + dt
    t += 1
    # ----

  rews = torch.hstack(rews)
  log_pis = torch.stack(log_pis).permute(1, 2, 0)
  vs = torch.vstack(vs).T
  actions = torch.vstack(actions).T
  s0s = torch.stack(s0s).T
  s1s = torch.stack(s1s).T
  pred_states = torch.stack(pred_states).permute(1, 2, 0)
  pred_rewards = torch.stack(pred_rewards).permute(1, 2, 0)
  true_rew_locs = rew_loc.tile(pred_rewards.shape[2], 1).T

  # update losses
  # ------------
  # δs = calc_deltas(rews[1, :, :], agent_outputs[Naction + 1, :, :]) #TD errors
  # Vterm = δs[:, t] .* agent_outputs[Naction + 1, :, t] #value function (batch)
  # L -= sum(loss_hp.βv * Vterm .* active) (sum over active episodes through multiplication by 'active')
  # RPE_term = δs[b, t] * agent_outputs[actions[1, b, t], b, t] #PG term
  # L -= loss_hp.βr * RPE_term
  # L += (loss_hp.βp * Lpred) #add predictive loss for internal world model
  # L -= loss_hp.βe * Lprior #add prior loss (formulated as a likelihood above)
  # L /= batch #normalize by batch
  # ------------

  #  jl: βp: 0.5 | βe: 0.05 | βv: 0.05 | βr: 1.0
  beta_p = 0.5  # predictive weight  
  beta_e = 0.05 # prior weight
  beta_v = 0.05 # value function weight
  beta_r = 1.   # reward prediction weight
  active = 1. # @TODO

  L_vt = torch.tensor(0.0, device=meta.device)
  L_rpe = torch.tensor(0.0, device=meta.device)
  L_pred = torch.tensor(0.0, device=meta.device)

  ds = td_error(rews, vs)
  N = ds.shape[1]
  for t in range(N):
    vt = ds[:, t] * vs[:, t] # value function (batch)
    vt_term = torch.sum(vt * active)
    L_vt -= vt_term

    # td errors weighted by logits of selected actions
    lp = torch.tensor([log_pis[i, actions[i, t], t] for i in range(log_pis.shape[0])]).to(meta.device)
    rpe_term = ds[:, t] * lp
    L_rpe -= torch.sum(rpe_term)

  # ------------
  state_pred_loss = state_prediction_loss(s0s, pred_states)
  reward_pred_loss = reward_prediction_loss(true_rew_locs, pred_rewards)
  L_pred += state_pred_loss + reward_pred_loss
  # ------------

  """
  jl: pred: 6203.166 | prior: -5.09006 | val: 8.010938 | rpe: -238.36337
  py: pred: 7123.762 | prior: ... | val: -14.416 | rpe: -45.529
  """
  # print(f'pred: {(L_pred*beta_p).item():.3f} | prior: ... | val: {(L_vt*beta_v).item():.3f} | rpe: {(L_rpe*beta_r).item():.3f}')

  L = L_vt * beta_v + L_rpe * beta_r + L_pred * beta_p

  print(f'Loss: {L.item()} | p(plan): {(p_plan/d_plan):.3f}')

  return L

def main():
  s = 4 # arena side length
  batch_size = 40
  num_episodes = 1000
  device = torch.device('cpu')

  meta = make_meta(arena_len=s, batch_size=batch_size, plan_len=8, device=device)
  model = build_model(meta=meta)
  mazes = [env.build_maze_arena(s) for _ in range(meta.batch_size)]

  optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

  for e in range(num_episodes):
    print(f'{e+1} of {num_episodes}')
    loss = run_episode(meta, model, mazes)
    optim.zero_grad()
    loss.backward()
    optim.step()

if __name__ == '__main__':
  main()