from model import AgentModel
from typing import Dict

def instantiate_model_from_checkpoint(sd: Dict[str, any]) -> AgentModel:
  mdl = AgentModel.from_ctor_params(sd['params'])
  mdl.load_state_dict(sd['state'])
  return mdl