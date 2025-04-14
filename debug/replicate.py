import sys; sys.path.append('..')
import eval
import env
from environment import MazeEnvironment
from evaluate import exploit_reward_state_prediction_accuracy_from_episode_result
from model import AgentModel
from debug_utility import put_params
from utility import dataclass_to_dict, filter_dict, split_array_indices, tensor_to_ndarray
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import numpy as np
import os
from itertools import product
from typing import List
import dataclasses
from dataclasses import dataclass
from multiprocessing import Process

NUM_PROCESSES = 5
# NUM_PROCESSES = 0

@dataclass
class Uniform:
  rep_dst_p = '/Users/nick/source/mattarlab/planning-network/results'
  cp_src_p = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models'
  do_save = True
  meta: eval.Meta
  ep_p: eval.EpisodeParams
  environ: MazeEnvironment
  hiddens: int

class ExploitTeleportIntervention(eval.EpisodeIntervention):
  def __init__(self, params: eval.EpisodeParams):
    super().__init__()
    self.batch_size = 0
    self.params = params

  def initialize(self, environ: MazeEnvironment):
    self.batch_size = environ.batch_size()
    self.is_exploit = torch.zeros((self.batch_size,), dtype=torch.bool)

  def begin_step(self):
    # import pdb; pdb.set_trace()
    pass

  def end_step(
    self, meta: eval.Meta, environ: MazeEnvironment, model: AgentModel, h_rnn, 
    pi, pred_output, s, s1, a, a1, prev_rewards, rew, time):
    """
    """
    got_reward = (rew > 0.).squeeze(1)
    first_exploit = np.argwhere(torch.logical_not(self.is_exploit) & got_reward).squeeze(0)
    self.is_exploit[got_reward] = True
    
    # teleport the agent to each state, compute the policy in the state with and without a rollout
    policies = np.zeros(0)
    policies_with_rollout = np.zeros(0)

    if first_exploit.numel() > 0:
      _, pred_reward = eval.decompose_prediction_output(meta, pred_output)

      policies = torch.zeros((first_exploit.numel(), environ.num_states, meta.num_actions))
      policies_with_rollout = torch.zeros(policies.shape)

      for si in range(environ.num_states):
        sp = s.clone()
        sp[:] = si

        # no rollout
        plan_input = None # @TODO
        x0 = eval.gen_input(
          meta=meta, environ=environ, prev_ahot=eval.one_hot_actions(a, meta.num_actions), 
          prev_rewards=prev_rewards, shot=F.one_hot(sp, meta.num_states), 
          time=time, plan_input=plan_input)
        
        h1, log_pi, v, pred_output, a1, _ = eval.forward_agent_model(
          meta=meta, model=model, x=x0, h_rnn=h_rnn, 
          num_ticks=None, 
          greedy_actions=self.params.sample_actions_greedily, 
          disable_rollouts=self.params.disable_rollouts)
        
        policies[:, si, :] = log_pi[first_exploit, :].exp()

        # with rollout
        # @TODO: should we use `pred_reward` from the canonical episode, or after teleporting?
        paths_hot, plan_found_reward, plan_h_rnn = eval.plan(
          meta=meta, model=model, environ=environ, pi=first_exploit, h_rnn=h_rnn, s=sp, a=a,
          time=time, pred_reward=pred_reward, prev_rewards=prev_rewards)
        
        plan_input = eval.gen_plan_input(meta=meta, path_hot=paths_hot, found_reward=plan_found_reward)
        x1 = eval.gen_input(
          meta=meta, environ=environ, prev_ahot=eval.one_hot_actions(a, meta.num_actions), 
          prev_rewards=prev_rewards, shot=F.one_hot(sp, meta.num_states), 
          time=time, plan_input=plan_input)
        
        h1, log_pi, v, pred_output, a1, _ = eval.forward_agent_model(
          meta=meta, model=model, x=x1, h_rnn=h_rnn, 
          num_ticks=None, 
          greedy_actions=self.params.sample_actions_greedily, 
          disable_rollouts=self.params.disable_rollouts)

        policies_with_rollout[:, si, :] = log_pi[first_exploit, :].exp()
      
    results = dict()
    results['selected'] = first_exploit
    results['policies'] = policies
    results['policies_with_rollout'] = policies_with_rollout
    results = {k: tensor_to_ndarray(v) for k, v in results.items()}
    return results

def compute_vf_error(rews: np.ndarray, vs: np.ndarray):
  cum_r = np.hstack((np.zeros((rews.shape[0], 1)), np.cumsum(rews[:, :-1], axis=1)))
  vt = np.sum(rews, axis=1, keepdims=True) - cum_r
  vf_err = np.abs(vs - vt)
  return vf_err

def eval_one(uniform: Uniform, model: AgentModel, fname: str):
  cp_p = os.path.join(uniform.cp_src_p, fname)
  cp = loadmat(cp_p)
  put_params(model, cp)

  exploit_intervention = ExploitTeleportIntervention(uniform.ep_p)
  base_p = dataclasses.replace(uniform.ep_p, interventions=[exploit_intervention])

  args = (uniform.meta, model, uniform.environ)
  res = eval.run_episode(*args, params=base_p)
  res_no_rollouts = eval.run_episode(*args, params=dataclasses.replace(base_p, disable_rollouts=True))

  variants = [res, res_no_rollouts]
  variant_info = [{'rollouts_enabled': True}, {'rollouts_enabled': False}]
  save_res = {'data': []}

  for i, res in enumerate(variants):
    resd = dataclass_to_dict(res)
    resd['vf_error'] = np.mean(compute_vf_error(resd['rewards'], resd['v']), axis=1)
    resd['exploit_reward_prediction_acc'] = exploit_reward_state_prediction_accuracy_from_episode_result(res)
    resd['variant_info'] = variant_info[i]
    resd['walls'] = np.stack([x.walls for x in uniform.environ.mazes])
    keys = ['p_plan', 'mean_total_reward', 'vf_error', 'actions', 'actives', 'rewards',
            'state_prediction_acc', 'reward_prediction_acc', 'exploit_reward_prediction_acc',
            'intervention_results', 'walls', 'reward_locs', 'states0', 'states1', 'variant_info']
    resd = filter_dict(resd, keys)
    save_res['data'].append(resd)
  
  if uniform.do_save:
    dst_p = os.path.join(uniform.rep_dst_p, f'replicate-{fname}')
    savemat(dst_p, save_res)

def eval_loop(uniform: Uniform, its: List):
  model = eval.build_model(
    meta=uniform.meta, hidden_size=uniform.hiddens, recurrent_layer_type='gru')
  model.eval()

  for i in range(its.shape[0]):
    print(f'\t{i+1} of {its.shape[0]}')
    seed, epoch = its[i, :]
    fname = f'N{uniform.hiddens}_T50_Lplan8_seed{seed}_{epoch}-dst.mat'
    eval_one(uniform, model, fname)

if __name__ == '__main__':
  hiddens = [60, 80, 100]
  seeds = [61, 62, 63, 64, 65]
  epochs = np.arange(11) * 100

  ep_p = eval.EpisodeParams(sample_actions_greedily=True)

  arena_len = 4
  num_episodes = 1_000
  meta = eval.make_meta(
    arena_len=arena_len, plan_len=8, device=torch.device('cpu'), 
    planning_enabled=True, 
    agent_chooses_ticks_enabled=False, 
    include_dummy_prediction_output=True,
    ticks_take_time=False)

  fixed_mazes = env.build_fixed_maze_arenas(arena_len, num_episodes)
  environ = MazeEnvironment(
    mazes=fixed_mazes, num_states=meta.num_states, 
    planning_action=meta.planning_action, device=meta.device)

  for ih, hid in enumerate(hiddens):
    print(f'{ih+1} of {len(hiddens)}')

    its = np.array([*product(seeds, epochs)], dtype=object)
    un = Uniform(meta=meta, ep_p=ep_p, environ=environ, hiddens=hid)

    if NUM_PROCESSES <= 0:
      eval_loop(un, its)
    else:
      pi = split_array_indices(len(its), NUM_PROCESSES)
      process_args = [(un, its[x, :]) for x in pi]
      processes = [Process(target=eval_loop, args=args) for args in process_args]
      for p in processes: p.start()
      for p in processes: p.join()
