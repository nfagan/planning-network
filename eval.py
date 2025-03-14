from model import AgentModel, RNN, Policy, Prediction
from utility import DBG
import env
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
from time import time as time_fn

# ----------------------------------------------------------------------------

@dataclass
class EpisodeResult:
  loss: torch.Tensor
  rewards: torch.Tensor
  actions: torch.Tensor
  actives: torch.Tensor
  states0: torch.Tensor
  states1: torch.Tensor
  log_pi: torch.Tensor
  forced_rollouts: torch.Tensor
  chosen_num_ticks: torch.Tensor
  v: torch.Tensor
  predicted_states: torch.Tensor
  predicted_rewards: torch.Tensor
  reward_locs: torch.Tensor
  mean_total_reward: float
  p_plan: float
  state_prediction_acc: float
  reward_prediction_acc: float

@dataclass
class EpisodeParams:
  NUM_TICKS_EXPLORE_ONLY = 1
  NUM_TICKS_EXPLOIT_ONLY = 2
  NUM_TICKS_ONCE_RANDOMLY = 3
  NUM_TICKS_APPLIES_ALWAYS = 4

  num_rollouts_per_planning_action: int = 1
  num_ticks_per_step: int = 1
  num_ticks_per_step_applies: int = 0
  num_ticks_per_step_is_randomized: bool = False
  force_rollouts_at_start_of_exploit_phase: bool = False
  sample_actions_greedily: bool = False
  verbose: int = 0

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
  T = 20.0 # time in trial
  concrete_action_time = 0.4
  planning_action_time = 0.12
  # ----
  #  jl: βp: 0.5 | βe: 0.05 | βv: 0.05 | βr: 1.0
  beta_p = 0.5  # predictive weight
  beta_e = 0.05 # prior weight
  # beta_e = 0.00 # prior weight
  beta_v = 0.05 # value function weight
  beta_r = 1.   # reward prediction weight

  agent_chooses_ticks_enabled: bool = False
  ticks_take_time: bool = False

# ----------------------------------------------------------------------------

def build_model(*, meta: Meta, hidden_size=100, recurrent_layer_type='gru') -> AgentModel:
  hs = hidden_size
  num_inputs, num_outputs, num_actions = meta.num_inputs, meta.num_outputs, meta.num_actions
  policy_output_size = num_actions + 1  # +1 for value
  # i.e., only predict state occupancy (ignore policy + value outputs)
  pred_output_size = num_outputs - num_actions - 1 

  rnn = RNN(in_size=num_inputs, hidden_size=hs, recurrent_layer_type=recurrent_layer_type)
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
  pred_s = pred_output[:, :meta.num_states]
  pred_r = pred_output[:, meta.num_states:]
  return pred_s, pred_r

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
  return torch.sum(targets[act] == ind[act]) / act.shape[0]

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
    num_ticks: torch.Tensor, greedy_actions: bool):
  """
  a2c.jl/forward_modular
  """
  def _policy_subset(log_pi_v): return log_pi_v[:, :meta.num_actions]
  def _v_subset(log_pi_v): return log_pi_v[:, meta.num_actions]

  max_num_ticks = torch.max(num_ticks)

  for i in range(max_num_ticks):
    if i == 0:
      h_rnn, ytemp = model.rnn(x, h_rnn)
    else:
      ri = i < num_ticks
      h_rnn[ri, :], ytemp[ri, :] = model.rnn(x[ri, :], h_rnn[ri, :])

  log_pi_v = model.policy(ytemp)
  log_pi = _policy_subset(log_pi_v)
  v = _v_subset(log_pi_v)
  ps = torch.softmax(log_pi, dim=1)
  a = sample_actions(ps, greedy_actions)
  ah = F.one_hot(a, meta.num_actions)
  pred_input = torch.hstack([ytemp, ah])
  pred_output = model.prediction(pred_input)
  return h_rnn, torch.log(ps), v, pred_output, a, num_ticks

def gen_plan_input(*, meta: Meta, path_hot: torch.Tensor, found_reward: torch.Tensor):
  x = torch.zeros((path_hot.shape[0], meta.num_planning_inputs))
  x[:, :meta.num_planning_inputs-1] = path_hot
  x[:, meta.num_planning_inputs-1] = found_reward
  return x

def gen_input(
    *, meta: Meta, arenas: List[env.Arena], prev_ahot: torch.Tensor, 
    prev_rewards: torch.Tensor, time: torch.Tensor, shot: torch.Tensor, 
    plan_input: torch.Tensor = None):
  """
  walls.jl/gen_input, get_wall_input
  """
  walls = torch.vstack(
    [torch.tensor(x.walls[:, :, [0, 2]].flatten(), device=meta.device) for x in arenas])
  x = torch.zeros((len(arenas), meta.num_inputs), device=meta.device)
  x[:, :meta.num_actions] = prev_ahot
  x[:, meta.num_actions:meta.num_actions+1] = prev_rewards
  x[:, meta.num_actions+1:meta.num_actions+2] = time / meta.T
  x[:, meta.num_actions+2:meta.num_actions+2+meta.num_states] = shot
  x[:, meta.num_actions+2+meta.num_states:meta.num_actions+2+meta.num_states+meta.num_states*2] = walls
  if plan_input is not None:
    x[:, meta.num_actions+2+meta.num_states+meta.num_states*2:] = plan_input
  return x

def rollout(
    *, meta: Meta, model: AgentModel, arenas: List[env.Arena], 
    h_rnn: torch.Tensor, s: torch.Tensor, a: torch.Tensor, goal_s: torch.Tensor,
    rewards: torch.Tensor, time: torch.Tensor):
  """
  model_planner.jl/model_tree_search
  """
  batch_size = len(arenas)

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
      ahot = F.one_hot(a, meta.num_actions)
      x = gen_input(meta=meta, arenas=arenas, prev_ahot=ahot, prev_rewards=rewards, time=time, shot=shot)
      h_rnn, ytemp = model.rnn(x, h_rnn)

    log_pi_v = model.policy(ytemp)
    # ignore thinking action
    log_pi = log_pi_v[:, :meta.num_concrete_actions]
    # sample next actions
    ps = torch.softmax(log_pi, dim=1)
    # @NOTE: prefer sampling ("imagined") actions during rollouts even when actions are selected 
    # greedily in the "real" environment. see model_planner.jl/model_tree_search
    a = sample_actions(ps, greedy_actions=False)
    ah = F.one_hot(a, meta.num_actions)

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
    *, meta: Meta, model: AgentModel, mazes: List[env.Arena], pi: torch.Tensor,
    h_rnn: torch.Tensor, s: torch.Tensor, a: torch.Tensor, time: torch.Tensor,
    pred_reward: torch.Tensor, prev_rewards: torch.Tensor):
  """
  """
  batch_size = len(mazes)
  paths_hot = torch.zeros((batch_size, meta.plan_len * meta.num_concrete_actions)).to(meta.device)
  plan_found_reward = torch.zeros((batch_size,)).to(meta.device)
  dst_h_rnn = h_rnn.clone()

  # indices of planning actions
  with torch.no_grad():
    path, found_rew, h = rollout(
      meta=meta, model=model, 
      arenas=[mazes[i] for i in pi], 
      s=s[pi], a=a[pi], 
      goal_s=predicted_goal_states(pred_reward)[pi],
      h_rnn=h_rnn[pi, :], 
      rewards=prev_rewards[pi, :], 
      time=time[pi, :]
    )
    paths_hot[pi, :] = path.flatten(1)
    plan_found_reward[pi] = found_rew
    dst_h_rnn[pi, :] = h
  return paths_hot, plan_found_reward, dst_h_rnn

def act_concretely(
  meta: Meta, mazes: List[env.Arena], s: torch.Tensor, a1: torch.Tensor, rew_loc: torch.Tensor):
  """
  Move the agent in the environment, if its chosen action is not the planning action, and receive
  reward if the goal state is reached.
  """
  # @NOTE: See Kris' walls_build.jl/act_and_receive_reward
  at_rew = s == rew_loc
  s1 = torch.zeros_like(s)
  rew = torch.zeros((len(mazes), 1)).to(meta.device)

  for i in range(len(mazes)):
    act = a1[i].item()
    if act == meta.planning_action: 
      s1[i] = s[i]
      continue
    # perform a concrete action (a movement)
    sn = env.move_agent(s[i].item(), act, mazes[i])
    if sn == rew_loc[i] and not at_rew[i]:  # newly reached the goal
      # -- teleport the agent to a new state that isn't the reward location
      while False:
        sn = torch.randint(0, meta.num_states, (1,))
        if sn != rew_loc[i]:
          break
      # --
      # update reward
      rew[i, 0] = 1.
    s1[i] = sn

  s1[at_rew] = teleport_from_reward(s[at_rew], meta.num_states, rew_loc[at_rew])
  return s1, rew, at_rew

def teleport_from_reward(s: torch.Tensor, n: int, rew_loc: torch.Tensor):
  i = s == rew_loc
  while torch.any(i):
    s[i] = torch.randint(0, n, (torch.sum(i),))
    i = s == rew_loc
  return s

def time_points_for_randomized_ticks(T: float, concrete_action_time: float, batch_size: int):
  max_num_ticks = int(T / concrete_action_time)
  num_ticks = torch.randint(0, max_num_ticks, (batch_size,)).type(torch.float32)
  return num_ticks * concrete_action_time

def run_episode(
    meta: Meta, model: AgentModel, mazes: List[env.Arena], 
    params: EpisodeParams = EpisodeParams()) -> EpisodeResult:
  """
  """
  assert params.num_ticks_per_step_applies >= 0 and params.num_ticks_per_step_applies <= 4, \
    'Unrecognized num_ticks_per_step_applies enumeration value.'
  assert params.num_rollouts_per_planning_action > 0, \
    'Expected at least 1 rollout per planning action'
  """
  """
  batch_size = len(mazes)

  wall_clock_t0 = time_fn()

  # initial actions, reward locations, and starting states.
  a = torch.randint(0, meta.num_actions, (batch_size,)).to(meta.device)
  rew_loc = torch.randint(0, meta.num_states, (batch_size,)).to(meta.device)
  s = torch.randint(0, meta.num_states, (batch_size,)).to(meta.device)
  # ensure agent does not begin at reward location
  s = teleport_from_reward(s, meta.num_states, rew_loc).to(meta.device)

  prev_rewards = torch.zeros((batch_size, 1)).to(meta.device)
  time = torch.zeros((batch_size, 1)).to(meta.device)
  h_rnn = model.rnn.make_h0(batch_size, meta.device)

  rand_tick_ts = time_points_for_randomized_ticks(
    meta.T, meta.concrete_action_time, batch_size).to(meta.device)
  performed_rand_ticks = torch.zeros((batch_size,), dtype=torch.bool, device=meta.device)

  # phases
  explore_phase_len = torch.ones((batch_size,), dtype=torch.long, device=meta.device) * -1
  is_exploit = torch.zeros((batch_size,), dtype=torch.bool, device=meta.device)

  actions = []
  s0s = []  # states at step t
  s1s = []  # states at step t+1
  log_pis = []
  rews = []
  vs = []
  pred_states = []
  pred_rewards = []
  actives = []
  forced_rollouts = []
  chosen_ticks = []

  t = 0
  n_plan = d_plan = 0.0
  while torch.any(time < meta.T):
    # steps that haven't yet exceeded the time limit
    is_active = time < meta.T

    # prepare model inputs for this step
    plan_input = None
    if meta.planning_enabled and t > 0:
      plan_input = gen_plan_input(meta=meta, path_hot=paths_hot, found_reward=plan_found_reward)

    x = gen_input(
      meta=meta, arenas=mazes, prev_ahot=F.one_hot(a, meta.num_actions), 
      prev_rewards=prev_rewards, shot=F.one_hot(s, meta.num_states), 
      time=time, plan_input=plan_input)
    
    # mask out finished episodes
    x[~is_active.squeeze(1), :] = 0.
    
    """
    (begin) determine the number of ticks of recurrent processing to perform
    """
    num_ticks_per_step = torch.ones((batch_size,), dtype=torch.long).to(meta.device)

    if params.num_ticks_per_step_applies == EpisodeParams.NUM_TICKS_ONCE_RANDOMLY:
      candidates_for_ticks = (t >= rand_tick_ts) & torch.logical_not(performed_rand_ticks)
      num_ticks_per_step[candidates_for_ticks] = \
        num_ticks_per_step[candidates_for_ticks] * params.num_ticks_per_step
      performed_rand_ticks[candidates_for_ticks] = True

    elif params.num_ticks_per_step_applies == EpisodeParams.NUM_TICKS_EXPLORE_ONLY:
      # only use `params.num_ticks_per_step` ticks during the explore phase.
      num_ticks_per_step[torch.logical_not(is_exploit)] *= params.num_ticks_per_step

    elif params.num_ticks_per_step_applies == EpisodeParams.NUM_TICKS_EXPLOIT_ONLY:
      if t > 0:
        # first_exploit is computed later (for the t+1 th time step), so only valid when t > 0
        if False:
          # only use `params.num_ticks_per_step` ticks at the start of the exploit phase.
          num_ticks_per_step[first_exploit] = params.num_ticks_per_step
        elif False:
          # use `params.num_ticks_per_step` ticks throughout the exploit phase.
          num_ticks_per_step[is_exploit] = params.num_ticks_per_step
        elif True:
          # use `params.num_ticks_per_step` ticks throughout the exploit phase, but only for the 
          # same number of steps as the explore phase.
          explore_lens = explore_phase_len[is_exploit]; assert torch.all(explore_lens >= 0)
          t_off = t - explore_lens; assert torch.all(t_off >= 0)
          permit_ticks = torch.argwhere(is_exploit).squeeze(1)[t_off < explore_lens]
          num_ticks_per_step[permit_ticks] *= params.num_ticks_per_step
        
    else:
      # no condition on when `num_ticks_per_step` applies
      if params.num_ticks_per_step_is_randomized:
        # use up to `params.num_ticks_per_step` ticks
        num_ticks_per_step[:] = torch.randint(1, params.num_ticks_per_step, (batch_size,))
      else:
        # always use `params.num_ticks_per_step` ticks
        num_ticks_per_step = num_ticks_per_step * params.num_ticks_per_step
    """
    (end) determine the number of ticks of recurrent processing to perform
    """

    """
    (begin) step the environment
    """
    # perform a step of recurrent processing
    h_rnn, log_pi, v, pred_output, a1, chosen_num_ticks = forward_agent_model(
      meta=meta, model=model, x=x, h_rnn=h_rnn, 
      num_ticks=num_ticks_per_step, greedy_actions=params.sample_actions_greedily)
    pred_state, pred_reward = decompose_prediction_output(meta, pred_output)

    # update the agent's state, when the chosen action is concrete.
    s1, rew, at_rew = act_concretely(meta, mazes, s, a1, rew_loc)

    # mask out finished episodes.
    rew *= is_active

    # check whether the agent newly received reward on this step.
    got_reward = (rew > 0.).squeeze(1)
    """
    (end) step the environment
    """

    """
    (begin) update phases
    """
    # episodes that just entered the exploit phase
    first_exploit = torch.logical_not(is_exploit) & got_reward

    # episodes that are are in the exploit phase
    is_exploit[got_reward] = True

    # for examples that are starting the exploit phase, mark the length of the explore phase.
    # t + 1 because we update time later.
    explore_phase_len[first_exploit] = t + 1
    """
    (end) update phases
    """

    """
    (begin) planning
    """
    forced_rollout = torch.zeros_like(first_exploit)
    pi = torch.tensor([], dtype=torch.long, device=meta.device) # "plan indices"
    if meta.planning_enabled:
      # implement rollouts
      if params.force_rollouts_at_start_of_exploit_phase:
        # plan when this is the start of the exploit phase
        plan_mask = first_exploit
        forced_rollout = first_exploit
      else:
        # plan when the chosen action is to plan, and so long as we're not at the reward location.
        # @NOTE: see Kris' model_planner.jl
        plan_mask = a1 == meta.planning_action & ~at_rew
      # numeric indices of episodes that will plan
      pi = torch.argwhere(plan_mask & is_active.squeeze(1)).squeeze(1)

      plan_h_rnn = h_rnn.clone()
      for _ in range(params.num_rollouts_per_planning_action):
        # execute a rollout
        paths_hot, plan_found_reward, plan_h_rnn = plan(
          meta=meta, model=model, mazes=mazes, pi=pi, h_rnn=plan_h_rnn, s=s, a=a, 
          time=time, pred_reward=pred_reward, prev_rewards=prev_rewards)
    """
    (end) planning
    """

    """
    (begin) agent chooses ticks
    """
    if meta.agent_chooses_ticks_enabled:
      # think when the chosen action is to think
      think_mask = a1 == meta.planning_action
      pi = torch.argwhere(think_mask & is_active.squeeze(1)).squeeze(1)
    """
    (end) agent chooses ticks
    """

    n_plan += pi.numel()
    d_plan += torch.sum(is_active).item()

    # increment time
    dt = torch.ones_like(time) * meta.concrete_action_time
    if meta.planning_enabled or meta.agent_chooses_ticks_enabled:
      dt[pi, :] = meta.planning_action_time
    if meta.ticks_take_time: 
      dt += ((chosen_num_ticks - 1.) * meta.planning_action_time).view(dt.shape)

    # push results
    rews.append(rew)
    log_pis.append(log_pi)
    vs.append(v)
    actions.append(a1)
    s0s.append(s)
    s1s.append(s1)
    pred_states.append(pred_state)
    pred_rewards.append(pred_reward)
    actives.append(is_active)
    forced_rollouts.append(forced_rollout)
    chosen_ticks.append(chosen_num_ticks)
    
    # prepare next iteration
    # if meta.agent_chooses_ticks_enabled: rew[pi] = prev_rewards[pi] # carry forward previous rewards
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
  actives = torch.stack(actives).permute(1, 0, 2).squeeze(2).type(pred_rewards.dtype)
  forced_rollouts = torch.stack(forced_rollouts).T
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
  p_plan = 0. if d_plan == 0 else n_plan/d_plan

  if params.verbose > 0:
    nt = torch.mean(torch.sum(chosen_ticks.type(actives.dtype) * actives, dim=1) / torch.sum(actives, dim=1))
    print(
      f'loss: {L.item():.3f} | p(plan): {p_plan:.3f} | ' + 
      f'rew: {tot_rew.item():.3f} | t: {wall_clock_t:.3f} | ticks: {nt.item():.2f}')

  return EpisodeResult(
    loss=L, rewards=rews, actions=actions, actives=actives,
    forced_rollouts=forced_rollouts,
    log_pi=log_pis, v=vs,
    mean_total_reward=tot_rew.item(), p_plan=p_plan, 
    state_prediction_acc=state_pred_acc.item(), 
    reward_prediction_acc=reward_pred_acc.item(),
    predicted_states=to_pred_state(pred_states), 
    predicted_rewards=to_pred_state(pred_rewards),
    reward_locs=rew_loc, states0=s0s, states1=s1s, chosen_num_ticks=chosen_ticks)