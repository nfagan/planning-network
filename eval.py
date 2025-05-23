from model import AgentModel, RNN, Policy, Prediction
from utility import DBG, _is_deterministic
from environment import Environment
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict
from time import time as time_fn

# ----------------------------------------------------------------------------

_NO_ACTION = -1

class EpisodeIntervention(object):
  def __init__(self): pass
  def initialize(self, environ: Environment): pass
  def begin_step(self): pass
  def end_step(self, *args): pass
  def transform_input(self, x: torch.Tensor): return x

@dataclass
class EpisodeResult:
  loss: torch.Tensor
  rewards: torch.Tensor
  actions: torch.Tensor
  actives: torch.Tensor
  states0: torch.Tensor
  states1: torch.Tensor
  xs: torch.Tensor
  hs: torch.Tensor
  log_pi: torch.Tensor
  chosen_num_ticks: torch.Tensor
  v: torch.Tensor
  predicted_states: torch.Tensor
  predicted_rewards: torch.Tensor
  reward_locs: torch.Tensor
  mean_total_reward: float
  p_plan: float
  state_prediction_acc: float
  reward_prediction_acc: float
  intervention_results: List[Dict]

@dataclass
class EpisodeParams:
  NUM_TICKS_EXPLORE_ONLY = 1
  NUM_TICKS_EXPLOIT_ONLY = 2
  NUM_TICKS_ONCE_RANDOMLY = 3
  NUM_TICKS_APPLIES_ALWAYS = 4

  num_rollouts_per_planning_action: int = 1
  disable_rollouts: bool = False
  num_ticks_per_step: int = 1
  num_ticks_per_step_applies: int = 0
  num_ticks_per_step_is_randomized: bool = False
  force_rollouts_at_start_of_exploit_phase: bool = False
  sample_actions_greedily: bool = False
  verbose: int = 0
  interventions: List[EpisodeIntervention] = field(default_factory=list)

@dataclass
class Meta:
  num_actions: int
  planning_enabled: bool
  num_concrete_actions: int # excludes planning action
  num_inputs: int
  num_planning_inputs: int
  num_planning_outputs: int
  num_outputs: int
  num_states: int
  plan_len: int
  planning_action = 4
  device: torch.device
  # ----
  # T = 20.0 # time in trial
  # concrete_action_time = 0.4
  # planning_action_time = 0.12
  # ----
  T = 50.0 # time in trial
  concrete_action_time = 1.0
  planning_action_time = 0.3
  # ----
  #  jl: βp: 0.5 | βe: 0.05 | βv: 0.05 | βr: 1.0
  beta_p = 0.5  # predictive weight
  # beta_p = 0.0  # predictive weight
  beta_e = 0.05 # prior weight
  # beta_e = 0.00 # prior weight
  beta_v = 0.05 # value function weight
  beta_r = 1.   # reward prediction weight

  agent_chooses_ticks_enabled: bool = False
  ticks_take_time: bool = False
  include_dummy_prediction_output: bool = False

# ----------------------------------------------------------------------------

def build_model(
  *, meta: Meta, hidden_size=100, recurrent_layer_type='gru', 
  use_zero_bias_hh=True, use_trainable_h0=False) -> AgentModel:
  """"""
  hs = hidden_size
  num_inputs, num_outputs, num_actions = meta.num_inputs, meta.num_outputs, meta.num_actions
  policy_output_size = num_actions + 1  # +1 for value
  # i.e., only predict state occupancy (ignore policy + value outputs)
  pred_output_size = num_outputs - num_actions - 1 

  # @NOTE: See: walls_build.jl/useful_dimensions
  if meta.include_dummy_prediction_output: pred_output_size += 1

  rnn = RNN(
    in_size=num_inputs, hidden_size=hs, 
    recurrent_layer_type=recurrent_layer_type, use_zero_bias_hh=use_zero_bias_hh,
    use_trainable_h0=use_trainable_h0
  )

  policy = Policy(rnn_hidden_size=hs, output_size=policy_output_size)
  prediction = Prediction(input_size=hs + num_actions, output_size=pred_output_size)
  return AgentModel(rnn=rnn, policy=policy, prediction=prediction).to(meta.device)

def make_meta(
    *, arena_len: int, plan_len: int, 
    device: torch.device, planning_enabled=True, 
    agent_chooses_ticks_enabled=False, ticks_take_time=False, **kwargs) -> Meta:
  """
  walls_build.jl/useful_dimensions
  planning.jl/build_planner
  """
  num_actions = 5 if (planning_enabled or agent_chooses_ticks_enabled) else 4
  num_concrete_actions = 4  # movements; excludes planning action
  num_states = arena_len * arena_len
  num_walls = 2 * num_states
  num_inputs = num_actions + 1 + 1 + num_states + num_walls # + 1 rew, + 1 time
  num_outputs = num_actions + 1 + num_states

  # (one-hot) action sequence and whether we ended at the reward location
  num_planning_inputs = 0 if not planning_enabled else 4 * plan_len + 1
  # (imagined) reward location
  num_planning_outputs = num_states

  num_inputs += num_planning_inputs
  num_outputs += num_planning_outputs

  m = Meta(
    num_inputs=num_inputs, num_outputs=num_outputs, 
    num_actions=num_actions, num_states=num_states, 
    plan_len=plan_len,
    num_concrete_actions=num_concrete_actions, 
    num_planning_inputs=num_planning_inputs, num_planning_outputs=num_planning_outputs,
    device=device, planning_enabled=planning_enabled, 
    agent_chooses_ticks_enabled=agent_chooses_ticks_enabled,
    ticks_take_time=ticks_take_time)
  for k in kwargs: setattr(m, k, kwargs[k])
  return m

def predicted_goal_states(pred_r: torch.Tensor):
  return torch.argmax(torch.softmax(pred_r, dim=1), dim=1)

def decompose_prediction_output(meta: Meta, pred_output: torch.Tensor):
  if meta.include_dummy_prediction_output:
    pred_s = pred_output[:, :meta.num_states]
    # Nout += 1 # needed for backward compatibility (this lives between state and reward predictions)
    pred_r = pred_output[:, meta.num_states+1:]
  else:
    pred_s = pred_output[:, :meta.num_states]
    pred_r = pred_output[:, meta.num_states:]
  return pred_s, pred_r

def one_hot_actions(a: torch.Tensor, num_actions: int, zero: int=_NO_ACTION) -> torch.Tensor:
  if False:
    return F.one_hot(a, num_actions)
  else:
    is_zero = a == zero
    if not is_zero.any(): return F.one_hot(a, num_actions)
    b = a + is_zero
    b = F.one_hot(b, num_actions)
    b[is_zero, :] = 0.
    return b

def to_pred_state(pred_s: torch.Tensor):
  logits = pred_s.permute(0, 2, 1)
  ind = torch.argmax(torch.softmax(logits, dim=2), dim=2)
  return ind

def prediction_loss(true_s: torch.Tensor, pred_s: torch.Tensor, actives: torch.Tensor):
  act = actives.flatten(0, 1).type(torch.bool)
  logits = pred_s.permute(0, 2, 1).flatten(0, 1)
  targets = true_s.flatten()
  return F.cross_entropy(logits[act], targets[act], reduction='sum')

def prediction_acc(true_s: torch.Tensor, pred_s: torch.Tensor, actives: torch.Tensor):
  act = actives.flatten(0, 1).type(torch.bool)
  logits = pred_s.permute(0, 2, 1).flatten(0, 1)
  targets = true_s.flatten()
  ind = torch.argmax(torch.softmax(logits, dim=1), dim=1)
  # import pdb; pdb.set_trace()
  # return torch.sum(targets[act] == ind[act]) / act.shape[0]
  return torch.sum(targets[act] == ind[act]) / act.sum()

def td_error(rews: torch.Tensor, v: torch.Tensor):
  N = rews.shape[1]
  r = torch.zeros((rews.shape[0],), device=v.device)
  ds = torch.zeros_like(v)
  for t in range(N):
    r = rews[:, N - t - 1] + r
    ds[:, N - t - 1] = r - v[:, N - t - 1]
  return ds

def sample_actions(ps: torch.Tensor, greedy_actions: bool): 
  if greedy_actions:
    return torch.argmax(ps, dim=1)
  else:
    res = torch.multinomial(ps, 1, True)
    return res.squeeze(1)

def prior_loss(meta: Meta, log_pi: torch.Tensor, active: torch.Tensor):
  """
  priors.jl: return -KL[q || p], q: log_pi, p: uniform
  """
  act = active.repeat(log_pi.shape[1], 1).T
  n_action = meta.num_actions
  logp = torch.log(torch.ones_like(log_pi) / n_action) * act
  log_pi = log_pi * act
  lprior = torch.sum(torch.exp(log_pi) * (logp - log_pi))
  return lprior

def forward_agent_model(
    *, meta: Meta, model: AgentModel, x: torch.Tensor, h_rnn: torch.Tensor, 
    num_ticks: torch.Tensor, greedy_actions: bool, disable_rollouts: bool):
  """
  a2c.jl/forward_modular
  """
  def _policy_subset(log_pi_v): return log_pi_v[:, :meta.num_actions]
  def _v_subset(log_pi_v): return log_pi_v[:, meta.num_actions]

  max_num_ticks = 1 if num_ticks is None else torch.max(num_ticks)
  for i in range(max_num_ticks):
    if i == 0:
      h_rnn, ytemp = model.rnn(x, h_rnn)
    else:
      ri = i < num_ticks
      h_rnn[ri, :], ytemp[ri, :] = model.rnn(x[ri, :], h_rnn[ri, :])

  log_pi_v = model.policy(ytemp)
  log_pi = _policy_subset(log_pi_v)
  v = _v_subset(log_pi_v)
  if disable_rollouts: log_pi = log_pi[:, :min(meta.num_actions, meta.num_concrete_actions)]
  ps = torch.softmax(log_pi, dim=1)
  a = sample_actions(ps, greedy_actions)
  ah = one_hot_actions(a, meta.num_actions)
  pred_input = torch.hstack([ytemp, ah])
  pred_output = model.prediction(pred_input)
  return h_rnn, torch.log(ps), v, pred_output, a, num_ticks

def gen_plan_input(*, meta: Meta, path_hot: torch.Tensor, found_reward: torch.Tensor):
  x = torch.zeros((path_hot.shape[0], meta.num_planning_inputs))
  x[:, :meta.num_planning_inputs-1] = path_hot
  x[:, meta.num_planning_inputs-1] = found_reward
  return x

def gen_input(
    *, meta: Meta, environ: Environment, prev_ahot: torch.Tensor, 
    prev_rewards: torch.Tensor, time: torch.Tensor, shot: torch.Tensor, 
    plan_input: torch.Tensor = None):
  """
  walls.jl/gen_input, get_wall_input
  """
  obs = environ.get_obs()
  x = torch.zeros((environ.batch_size(), meta.num_inputs), device=meta.device)
  x[:, :meta.num_actions] = prev_ahot
  x[:, meta.num_actions:meta.num_actions+1] = prev_rewards
  # x[:, meta.num_actions+1:meta.num_actions+2] = time / meta.T
  x[:, meta.num_actions+1:meta.num_actions+2] = time / 50.0; assert meta.T <= 50.0 # walls.jl/gen_input
  x[:, meta.num_actions+2:meta.num_actions+2+meta.num_states] = shot
  x[:, meta.num_actions+2+meta.num_states:meta.num_actions+2+meta.num_states+meta.num_states*2] = obs
  if plan_input is not None:
    x[:, meta.num_actions+2+meta.num_states+meta.num_states*2:] = plan_input
  return x

def rollout(
    *, meta: Meta, model: AgentModel, environ: Environment, 
    h_rnn: torch.Tensor, s: torch.Tensor, a: torch.Tensor, goal_s: torch.Tensor,
    rewards: torch.Tensor, time: torch.Tensor):
  """
  model_planner.jl/model_tree_search
  """
  batch_size = environ.batch_size()

  paths_hot = torch.zeros((batch_size, meta.num_concrete_actions, meta.plan_len), device=meta.device)
  found_rew = torch.zeros((batch_size,), device=meta.device)

  # remaining indices of rollouts that haven't yet reached the goal state.
  rmi = torch.ones((batch_size,), dtype=torch.bool, device=meta.device)

  ytemp = h_rnn
  dst_h_rnn = h_rnn.clone()

  for pi in range(meta.plan_len):
    if rmi.sum() == 0: break
    # numeric remaining indices (instead of boolean)
    nmri = torch.argwhere(rmi).squeeze(1)

    # for the first planning iteration, the policy derives from the rnn's current hidden state;
    # afterwards, the policy derives from the rnn's output.
    if pi > 0:
      shot = F.one_hot(s, meta.num_states)
      ahot = one_hot_actions(a, meta.num_actions)
      x = gen_input(meta=meta, environ=environ, prev_ahot=ahot, prev_rewards=rewards, time=time, shot=shot)
      h_rnn, ytemp = model.rnn(x, h_rnn)

    log_pi_v = model.policy(ytemp)
    # ignore thinking action
    log_pi = log_pi_v[:, :meta.num_concrete_actions]
    # sample next actions
    ps = torch.softmax(log_pi, dim=1)
    # @NOTE: prefer sampling ("imagined") actions during rollouts even when actions are selected 
    # greedily in the "real" environment. see model_planner.jl/model_tree_search
    a = sample_actions(ps, greedy_actions=True if _is_deterministic() else False)
    ah = one_hot_actions(a, meta.num_actions)

    # record chosen actions
    assert a.shape[0] == batch_size
    for i in range(nmri.shape[0]):
      paths_hot[nmri[i], a[nmri[i]], pi] = 1.

    # record hidden state of non-goal-reached rollouts
    dst_h_rnn[rmi, :] = h_rnn[rmi, :]

    # predict and sample new states
    pred_output = model.prediction(torch.hstack([ytemp, ah]))
    s = torch.argmax(torch.softmax(pred_output[:, :meta.num_states], dim=1), dim=1)

    # check whether we reached the goal, and only consider remaining (non-goal-reached) rollouts
    goal_reached = s == goal_s
    # mark that we found reward, but only consider remaining active episodes (rmi)
    found_rew[rmi & goal_reached] = 1
    rmi[goal_reached] = False

    time = time + 1.

  return paths_hot, found_rew, dst_h_rnn

def plan(
    *, meta: Meta, model: AgentModel, environ: Environment, pi: torch.Tensor,
    h_rnn: torch.Tensor, s: torch.Tensor, a: torch.Tensor, time: torch.Tensor,
    pred_reward: torch.Tensor, prev_rewards: torch.Tensor):
  """
  """
  batch_size = environ.batch_size()
  paths_hot = torch.zeros((batch_size, meta.plan_len * meta.num_concrete_actions)).to(meta.device)
  plan_found_reward = torch.zeros((batch_size,)).to(meta.device)
  dst_h_rnn = h_rnn.clone()

  # indices of planning actions
  with torch.no_grad():
    path, found_rew, h = rollout(
      meta=meta, model=model, 
      environ=environ.slice(pi),
      s=s[pi], a=a[pi], 
      goal_s=predicted_goal_states(pred_reward)[pi],
      h_rnn=h_rnn[pi, :], 
      rewards=prev_rewards[pi, :], 
      time=time[pi, :]
    )
    paths_hot[pi, :] = path.transpose(1, 2).flatten(1)
    plan_found_reward[pi] = found_rew
    dst_h_rnn[pi, :] = h
  return paths_hot, plan_found_reward, dst_h_rnn

def run_episode(
    meta: Meta, model: AgentModel, environ: Environment, 
    params: EpisodeParams = EpisodeParams()) -> EpisodeResult:
  """
  """
  assert params.num_ticks_per_step_applies >= 0 and params.num_ticks_per_step_applies <= 4, \
    'Unrecognized num_ticks_per_step_applies enumeration value.'
  assert params.num_rollouts_per_planning_action > 0, \
    'Expected at least 1 rollout per planning action'
  if meta.agent_chooses_ticks_enabled: assert False
  """
  """
  batch_size = environ.batch_size()

  wall_clock_t0 = time_fn()

  # initial actions, reward locations, and starting states.
  s, a = environ.initialize()

  prev_rewards = torch.zeros((batch_size, 1)).to(meta.device)
  time = torch.ones((batch_size, 1)).to(meta.device)  # @NOTE: see initializations.jl
  h_rnn = model.rnn.make_h0(batch_size, meta.device)

  for intervention in params.interventions: intervention.initialize(environ)

  actions = []
  s0s = []  # states at step t
  s1s = []  # states at step t+1
  log_pis = []
  rews = []
  vs = []
  xs = []
  hs = []
  pred_states = []
  pred_rewards = []
  actives = []
  chosen_ticks = []
  intervention_results = []

  t = 0
  # T_thresh = meta.T
  T_thresh = meta.T + 1.0 - 1e-2
  while torch.any(time < T_thresh):
    # begin intervention
    for intervention in params.interventions: intervention.begin_step()

    # steps that haven't yet exceeded the time limit
    is_active = time < T_thresh
    hs.append(h_rnn)

    # prepare model inputs for this step
    plan_input = None
    if meta.planning_enabled and t > 0:
      plan_input = gen_plan_input(meta=meta, path_hot=paths_hot, found_reward=plan_found_reward)

    x = gen_input(
      meta=meta, environ=environ, prev_ahot=one_hot_actions(a, meta.num_actions), 
      prev_rewards=prev_rewards, shot=F.one_hot(s, meta.num_states), 
      time=time, plan_input=plan_input)
    
    # intervene on inputs
    for intervention in params.interventions: x = intervention.transform_input(x)
    
    # mask out finished episodes
    x[~is_active.squeeze(1), :] = 0.
    xs.append(x)
    
    # number of ticks of recurrent processing to perform
    num_ticks_per_step = torch.ones((batch_size,), dtype=torch.long).to(meta.device)

    """
    (begin) step the environment
    """
    # perform a step of recurrent processing
    h_rnn, log_pi, v, pred_output, a1, chosen_num_ticks = forward_agent_model(
      meta=meta, model=model, x=x, h_rnn=h_rnn, 
      num_ticks=num_ticks_per_step, 
      greedy_actions=params.sample_actions_greedily, disable_rollouts=params.disable_rollouts)
    pred_state, pred_reward = decompose_prediction_output(meta, pred_output)

    # update the agent's state, when the chosen action is concrete.
    s1, s1p, rew, at_rew = environ.act(s, a1)

    # mask out finished episodes.
    rew *= is_active
    """
    (end) step the environment
    """

    """
    (begin) planning
    """
    pi = torch.tensor([], dtype=torch.long, device=meta.device) # "plan indices"
    if meta.planning_enabled:
      # implement rollouts
      if params.force_rollouts_at_start_of_exploit_phase:
        assert False
      else:
        # plan when the chosen action is to plan, and so long as we're not at the reward location.
        # @NOTE: see Kris' model_planner.jl
        plan_mask = (a1 == meta.planning_action) & ~at_rew
      # numeric indices of episodes that will plan
      pi = torch.argwhere(plan_mask & is_active.squeeze(1)).squeeze(1)

      plan_h_rnn = h_rnn.clone()
      for _ in range(params.num_rollouts_per_planning_action):
        # execute a rollout
        paths_hot, plan_found_reward, plan_h_rnn = plan(
          meta=meta, model=model, environ=environ, pi=pi, h_rnn=plan_h_rnn, s=s, a=a, 
          time=time, pred_reward=pred_reward, prev_rewards=prev_rewards)
    """
    (end) planning
    """

    # increment time
    dt = torch.ones_like(time) * meta.concrete_action_time
    if meta.planning_enabled or meta.agent_chooses_ticks_enabled:
      dt[pi, :] = meta.planning_action_time
    if meta.ticks_take_time: 
      dt += ((chosen_num_ticks - 1.) * meta.planning_action_time).view(dt.shape)

    intervene_result = []
    for intervention in params.interventions: 
      intervene_result.append(intervention.end_step(
        meta, environ, model, h_rnn, pi, pred_output, s, s1, a, a1, prev_rewards, rew, time))

    # push results
    rews.append(rew)
    log_pis.append(log_pi)
    vs.append(v)
    actions.append(a1)
    s0s.append(s)
    # s1s.append(s1)
    s1s.append(s1p)
    pred_states.append(pred_state)
    pred_rewards.append(pred_reward)
    actives.append(is_active)
    chosen_ticks.append(chosen_num_ticks)
    intervention_results.append(intervene_result)
    
    # prepare next iteration
    # if meta.agent_chooses_ticks_enabled: rew[pi] = prev_rewards[pi] # carry forward previous rewards
    a = a1
    s = s1
    prev_rewards = rew
    time = time + dt
    t += 1
    # ----

  rew_loc = environ.reward_location # @TODO: This is currently specific to the maze environment.
  rews = torch.hstack(rews)
  log_pis = torch.stack(log_pis).permute(1, 2, 0)
  vs = torch.vstack(vs).T
  actions = torch.vstack(actions).T
  s0s = torch.stack(s0s).T
  s1s = torch.stack(s1s).T
  xs = torch.stack(xs).permute(1, 2, 0)
  hs = torch.stack(hs).permute(1, 2, 0)
  pred_states = torch.stack(pred_states).permute(1, 2, 0)
  pred_rewards = torch.stack(pred_rewards).permute(1, 2, 0)
  true_rew_locs = rew_loc.tile(pred_rewards.shape[2], 1).T
  actives = torch.stack(actives).permute(1, 0, 2).squeeze(2).type(pred_rewards.dtype)
  chosen_ticks = torch.vstack(chosen_ticks).T

  # update losses
  # ------------
  beta_p = meta.beta_p  # predictive weight  
  beta_e = meta.beta_e  # prior weight
  beta_v = meta.beta_v  # value function weight
  beta_r = meta.beta_r  # reward prediction weight

  L_pred = torch.tensor(0.0, device=meta.device)
  L_prior = torch.tensor(0.0, device=meta.device)
  L_vt = torch.tensor(0.0, device=meta.device)
  L_rpe = torch.tensor(0.0, device=meta.device)

  with torch.no_grad():
    ds = td_error(rews, vs)

  """
  compute terms of the loss. elements are accumulated over steps within an episode and then averaged
  over episodes
  """
  N = ds.shape[1]
  for t in range(N):
    vt = ds[:, t] * vs[:, t] # value function (batch)
    vt_term = torch.sum(vt * actives[:, t])
    L_vt -= vt_term

    # td errors weighted by logits of selected actions
    lp = log_pis[torch.arange(log_pis.shape[0]), actions[:, t], t]

    rpe_term = ds[:, t].detach() * lp * actives[:, t]
    L_rpe -= torch.sum(rpe_term)

    # prior loss to encourage entropy in policy
    L_prior -= prior_loss(meta, log_pis[:, :, t], actives[:, t])

  # ------------
  state_pred_loss = prediction_loss(s1s, pred_states, actives)
  reward_pred_loss = prediction_loss(true_rew_locs, pred_rewards, actives)
  state_pred_acc = prediction_acc(s1s, pred_states, actives)
  reward_pred_acc = prediction_acc(true_rew_locs, pred_rewards, actives)
  L_pred += state_pred_loss + reward_pred_loss
  # ------------

  """
  jl: pred: 6203.166 | prior: -5.09006 | val: 8.010938 | rpe: -238.36337
  py: pred: 7123.762 | prior: ... | val: -14.416 | rpe: -45.529
  """

  L = L_vt * beta_v + L_rpe * beta_r + L_pred * beta_p + L_prior * beta_e
  L /= rews.shape[0]

  if params.verbose > 1:
    print(
      f'L: {L.item():.3f} | pred: {(L_pred*beta_p).item():.3f} | prior: {(L_prior*beta_e).item():.3f} | ' + 
      f'val: {(L_vt*beta_v).item():.3f} | rpe: {(L_rpe*beta_r).item():.3f}')

  # ------------
  tot_rew = torch.mean(torch.sum(rews * actives, dim=1))

  # ------------
  wall_clock_t = time_fn() - wall_clock_t0

  # ------------
  p_plan = (((actions == meta.planning_action) & (actives > 0)).sum() / actives.sum()).item()

  if params.verbose > 0:
    nt = torch.mean(torch.sum(chosen_ticks.type(actives.dtype) * actives, dim=1) / torch.sum(actives, dim=1))
    print(
      f'loss: {L.item():.3f} | p(plan): {p_plan:.3f} | ' + 
      f'rew: {tot_rew.item():.3f} | t: {wall_clock_t:.3f} | ticks: {nt.item():.2f}')

  return EpisodeResult(
    loss=L, rewards=rews, actions=actions, actives=actives,
    log_pi=log_pis, v=vs,
    mean_total_reward=tot_rew.item(), p_plan=p_plan, 
    state_prediction_acc=state_pred_acc.item(), 
    reward_prediction_acc=reward_pred_acc.item(),
    predicted_states=to_pred_state(pred_states), 
    predicted_rewards=to_pred_state(pred_rewards),
    reward_locs=rew_loc, states0=s0s, states1=s1s, xs=xs, hs=hs, chosen_num_ticks=chosen_ticks,
    intervention_results=intervention_results)