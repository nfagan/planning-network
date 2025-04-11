from model import AgentModel
from typing import Dict, List
import numpy as np
import torch

# hacky debug command. insert `exec(DBG)` into some scope to step into a REPL console at that point.
DBG = 'import code; __ctx001__ = globals().copy(); __ctx001__.update(locals()); code.interact(local=__ctx001__);'

def instantiate_model_from_checkpoint(sd: Dict[str, any]) -> AgentModel:
  mdl = AgentModel.from_ctor_params(sd['params'])
  mdl.load_state_dict(sd['state'])
  return mdl

def split_array_indices(M: int, N: int) -> List[np.ndarray[int]]:
  # Return indices to split sequence with length `M` into at most `N` disjoint sets
  if M >= N:
    subset_size = M // N
    subsets = []
    for i in range(N):
      start = i * subset_size
      end = (i + 1) * subset_size if i + 1 < N else M
      subsets.append(np.array(range(start, end)))
    return subsets
  else:
    # Not enough elements to form N subsets; return one-element subsets
    return [np.array(range(i, i+1)) for i in range(M)]
  
def filter_dict(d: Dict, ks: List[str]) -> Dict:
  r = {}
  for k in d.keys():
    if k in ks:
      r[k] = d[k]
  return r
  
def dataclass_to_dict(res, sel: List[str]=None):
  ks = dir(res)
  r = {}
  for k in ks:
    if sel is not None and k not in sel: continue
    if not k[0] == '_':
      v = getattr(res, k)
      if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
      r[k] = v
  return r