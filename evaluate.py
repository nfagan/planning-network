from model import AgentModel
from utility import instantiate_model_from_checkpoint, split_array_indices
import eval
import env
import torch
import numpy as np
from typing import List, Tuple
import os
from dataclasses import dataclass
from multiprocessing import Process

@dataclass
class Context:
  dst_dir = os.path.join(os.getcwd(), 'results')
  cp_root_dir = os.path.join(os.getcwd(), 'checkpoints')
  save = True
  num_processes = 8
  batch_size: int
  arena_len: int
  mazes: List[env.Arena]
  ep_p: eval.EpisodeParams

def load_checkpoint(cp_p: str) -> Tuple[AgentModel, eval.Meta, List[env.Arena]]:
  sd = torch.load(cp_p)
  model = instantiate_model_from_checkpoint(sd)
  meta = sd['meta']
  return model, meta, sd['mazes']

def compute_value_function_error(rews: torch.Tensor, vs: torch.Tensor):
  raise NotImplementedError

def find_rewards(rews: torch.Tensor):
  trials = []
  for i in range(rews.shape[0]):
    ri = torch.argwhere(rews[i, :]).squeeze(1)
    if ri.numel() > 0:
      trials.append([i.item() for i in ri])
    else:
      trials.append([])
  return trials

def find_first_exploit(rews: List[List[int]]):
  exploit_ind = np.ones((len(rews),), dtype=int) * -1
  for i, t in enumerate(rews):
    if len(t) > 1:
      exploit_ind[i] = t[1]
  return exploit_ind

def exploit_reward_state_prediction_accuracy(
  exploit_ind: np.ndarray, rew_locs: torch.Tensor, pred_rew: torch.Tensor):
  #
  n = 0
  d = 0
  for i in range(exploit_ind.shape[0]):
    ei = exploit_ind[i]
    if ei < 0: continue
    rew = rew_locs[i]
    pred = pred_rew[i, ei:]
    n += torch.sum(rew == pred).item()
    d += pred.numel()
  return 0. if d == 0 else n/d

def evaluate_forced_num_ticks(model: AgentModel, meta: eval.Meta, mazes: List[env.Arena]):
  batch_size = len(mazes)

  num_ticks = [*range(1, 9)]
  mean_rew = torch.zeros((batch_size, len(num_ticks))) + torch.nan

  for it in range(len(num_ticks)):
    ep_p = eval.EpisodeParams(
      num_ticks_per_step=num_ticks[it], 
      num_ticks_per_step_only_applies_at_start_of_exploit_phase=False
    )
    res = eval.run_episode(meta, model, mazes, params=ep_p)
    rews = torch.sum(res.rewards * res.actives, dim=1)
    mean_rew[:, it] = rews

  return np.array(num_ticks), mean_rew.detach().cpu().numpy()

# -------------------------------------------------------------------------------------

def evaluate_forced_rollouts(model: AgentModel, meta: eval.Meta, mazes: List[env.Arena]):
  batch_size = len(mazes)

  num_rollouts = [*range(1, 12)]
  entropies = torch.zeros((batch_size, len(num_rollouts))) + torch.nan

  for it in range(len(num_rollouts)):
    ep_p = eval.EpisodeParams(
      num_rollouts_per_planning_action=num_rollouts[it], 
      force_rollouts_at_start_of_exploit_phase=True,
      verbose=0
    )

    res = eval.run_episode(meta, model, mazes, params=ep_p)

    for i in range(batch_size):
      forced_rollout = torch.argwhere(res.forced_rollouts[i, :]).squeeze()
      if forced_rollout.numel() == 0: continue
      # rollouts are input to the network on the subsequent iteration
      result_of_rollout_ind = forced_rollout.item() + 1
      if result_of_rollout_ind >= res.log_pi.shape[2]: continue
      # renormalize over number of concrete actions
      log_p = res.log_pi[i, :meta.num_concrete_actions, result_of_rollout_ind]
      ps = torch.softmax(log_p, dim=0)
      entropies[i, it] = -torch.sum(ps * torch.log(ps))

  return np.array(num_rollouts), entropies.detach().cpu().numpy()

# -------------------------------------------------------------------------------------

def evaluate_one(
  ctx: Context, model: AgentModel, meta: eval.Meta, cp_mazes: List[env.Arena]):
  """
  """
  res = eval.run_episode(meta, model, ctx.mazes, params=ctx.ep_p)
  train_res = eval.run_episode(meta, model, cp_mazes, params=ctx.ep_p)

  ri = find_rewards(res.rewards)
  first_exploit = find_first_exploit(ri)
  exploit_acc = exploit_reward_state_prediction_accuracy(
    first_exploit, res.reward_locs, res.predicted_rewards)
  # v_errors = compute_value_function_error(res.rewards, res.v)
  
  num_ticks, forced_ticks_mean_rew = evaluate_forced_num_ticks(model, meta, ctx.mazes)
  num_entropy_rollouts, policy_entropies = evaluate_forced_rollouts(model, meta, ctx.mazes)
  
  row = {
    'res': res,
    'train_res': train_res,
    'first_exploit': first_exploit,
    'exploit_acc': exploit_acc,
    'num_forced_rollouts': num_entropy_rollouts,
    'forced_rollout_policy_entropies': policy_entropies,
    'num_ticks': num_ticks,
    'forced_ticks_mean_reward': forced_ticks_mean_rew,
  }

  return row

def evaluate_n(ctx: Context, subdir: str, cp_inds: List[int], tot_experience: List[int]):
  for i in range(len(cp_inds)):
    print(f'\t{i+1} of {len(cp_inds)}')

    cp_f = f'cp-{cp_inds[i]}.pth'
    cp_p = os.path.join(os.path.join(ctx.cp_root_dir, subdir, cp_f))

    model, meta, cp_mazes = load_checkpoint(cp_p)

    row = evaluate_one(ctx, model, meta, cp_mazes)
    row['subdirectory'] = subdir
    row['experience'] = tot_experience[i]

    if ctx.save:
      torch.save({'row': row}, os.path.join(ctx.dst_dir, f'evaluation-{subdir}-{cp_f}'))

def evaluate():
  arena_len = 4
  batch_size = int(1e3)

  ctx = Context(
    batch_size=batch_size,
    arena_len=arena_len,
    mazes=env.build_maze_arenas(arena_len, batch_size),
    ep_p=eval.EpisodeParams(verbose=0),
  )

  cp_subdirs = [
    # 'plan-yes-full-short-rollouts',
    # 'plan-yes-full',
    # 'plan-yes-full-60',
    # 'plan_no-hs_100-plan_len_8',
    'plan_no-hs_100-plan_len_8-rand_ticks_yes-num_ticks_16'
  ]

  cp_ind_sets = [
    np.array([*np.arange(0, int(195e3)+1, int(5e3)), int(200e3 - 1)])
  ] * len(cp_subdirs)

  for si, subdir in enumerate(cp_subdirs):
    print(f'{subdir} ({si+1} of {len(cp_subdirs)})')

    cp_inds = cp_ind_sets[si]
    tot_experience = cp_inds * 40 # @TODO: This batch size was fixed during training

    if ctx.num_processes <= 0:
      # no multiprocessing
      evaluate_n(ctx, subdir, cp_inds, tot_experience)
    else:
      pi = split_array_indices(len(cp_inds), ctx.num_processes)
      process_args = [(ctx, subdir, cp_inds[pi[i]], tot_experience[pi[i]]) for i in range(len(pi))]
      processes = [Process(target=evaluate_n, args=args) for args in process_args]
      for p in processes: p.start()
      for p in processes: p.join()

if __name__ == '__main__':
  evaluate()