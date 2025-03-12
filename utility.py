from model import AgentModel
from typing import Dict, List
import numpy as np

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