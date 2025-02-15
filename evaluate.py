from model import AgentModel
from utility import instantiate_model_from_checkpoint
import eval
import env
import torch
import numpy as np
from typing import List, Tuple
import os

def load_checkpoint(cp_p: str) -> Tuple[AgentModel, eval.Meta, List[env.Arena]]:
  sd = torch.load(cp_p)
  model = instantiate_model_from_checkpoint(sd)
  meta = sd['meta']
  return model, meta, sd['mazes']

def compute_value_function_error(rews: torch.Tensor, vs: torch.Tensor):
  import pdb; pdb.set_trace()

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

# -------------------------------------------------------------------------------------

def evaluate_trained():
  save = True
  batch_size = int(1e3)
  arena_len = 4
  mazes = env.build_maze_arenas(arena_len, batch_size)

  subdir = 'plan-yes-full-short-rollouts'
  cp_dir = os.path.join(os.getcwd(), 'checkpoints', subdir)
  dst_dir = os.path.join(os.getcwd(), 'results')

  # cp_inds = np.arange(0, int(140e3)+1, int(5e3))
  cp_inds = np.arange(0, int(195e3)+1, int(5e3))
  cp_inds = np.array([*cp_inds, int(200e3 - 1)])
  ep_p = eval.EpisodeParams(verbose=0)

  tot_experience = cp_inds * 40 # @TODO: This batch size was fixed during training

  for i in range(len(cp_inds)):
    print(f'{i+1} of {len(cp_inds)}')

    cp_f = f'cp-{cp_inds[i]}.pth'
    cp_p = os.path.join(cp_dir, cp_f)
    model, meta, cp_mazes = load_checkpoint(cp_p)

    res = eval.run_episode(meta, model, mazes, params=ep_p)
    train_res = eval.run_episode(meta, model, cp_mazes, params=ep_p)

    ri = find_rewards(res.rewards)
    first_exploit = find_first_exploit(ri)
    exploit_acc = exploit_reward_state_prediction_accuracy(
      first_exploit, res.reward_locs, res.predicted_rewards)
    # v_errors = compute_value_function_error(res.rewards, res.v)
    
    num_entropy_rollouts, policy_entropies = evaluate_policy_entropy(model, meta, mazes)
    
    row = {
      'res': res,
      'train_res': train_res,
      'first_exploit': first_exploit,
      'exploit_acc': exploit_acc,
      'experience': tot_experience[i],
      'num_entropy_rollouts': num_entropy_rollouts,
      'policy_entropies': policy_entropies,
      'subdirectory': subdir
    }

    if save:
      torch.save({'row': row}, os.path.join(dst_dir, f'evaluation-{subdir}-{cp_f}'))

# -------------------------------------------------------------------------------------

def evaluate_policy_entropy(model: AgentModel, meta: eval.Meta, mazes: List[env.Arena]):
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

def do_evaluate_policy_entropy():
  subdir = 'plan-yes-full'
  cp_dir = os.path.join(os.getcwd(), 'checkpoints', subdir)

  cp_ind = int(200e3 - 1)
  cp_f = f'cp-{cp_ind}.pth'
  cp_p = os.path.join(cp_dir, cp_f)
  model, meta, _ = load_checkpoint(cp_p)

  num_rollouts, entropies = evaluate_policy_entropy(model, meta, env.build_maze_arenas(4, int(1e3)))

if __name__ == '__main__':
  evaluate_trained()
  # do_evaluate_policy_entropy()