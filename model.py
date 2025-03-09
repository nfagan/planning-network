import torch
import torch.nn as nn
from typing import Dict

class RNN(nn.Module):
  def __init__(self, *, in_size: int, hidden_size: int, recurrent_layer_type='gru'):
    """
    """
    assert in_size > 0 and hidden_size > 0
    assert recurrent_layer_type in ['gru', 'rnn'], f'Unrecognized layer type {recurrent_layer_type}'

    super().__init__()

    self.in_size = in_size
    self.hidden_size = hidden_size
    self.recurrent_layer_type = recurrent_layer_type

    if recurrent_layer_type == 'gru':
      self.gru = nn.GRU(in_size, hidden_size, batch_first=True)
    elif recurrent_layer_type == 'rnn':
      self.gru = nn.RNN(in_size, hidden_size, batch_first=True)
    else:
      raise NotImplementedError

  def make_h0(self, batch_size: int, device: torch.device):
    return torch.zeros((batch_size, self.hidden_size), device=device)

  def forward(self, x: torch.Tensor, h: torch.Tensor):
    """
    Inputs consisted of the current agent location sk, previous action a{k - 1}, reward r{k - 1}, 
    wall locations and the elapsed time t since the start of the episode (Methods)
    ...
    x: (batch, in_size)
    h: (batch, hidden_size)
    """
    # (batch, sequence_length[=1], hidden or input size)
    xh, xw = x.shape[0], x.shape[1]
    hh, hw = h.shape[0], h.shape[1]
    y, h1 = self.gru(x.reshape(xh, 1, xw), h.reshape(1, hh, hw))
    return y.squeeze(1), h1.squeeze(0)
  
  def ctor_params(self):
    return {
      'in_size': self.in_size, 'hidden_size': self.hidden_size, 
      'recurrent_layer_type': self.recurrent_layer_type
    }

class Policy(nn.Module):
  def __init__(self, *, rnn_hidden_size: int, output_size: int):
    """
    model.jl:
    policy = Chain(Dense(mp.Nhidden, Naction+1)) #policy and value function
    """
    super().__init__()
    assert rnn_hidden_size > 0 and output_size > 0
    self.rnn_hidden_size = rnn_hidden_size
    self.output_size = output_size
    self.linear = nn.Linear(rnn_hidden_size, output_size)

  def forward(self, y: torch.Tensor):
    """
    Note that we do not have a nonlinearity here because the output is interpreted to contain both a 
    policy and value function. In Kris' implementation, the softmax is computed later for the subset
    of components representing the policy; this is in the function `forward_modular`
    """
    return self.linear(y)
  
  def ctor_params(self):
    return {'rnn_hidden_size': self.rnn_hidden_size, 'output_size': self.output_size}
  
class Prediction(nn.Module):
  def __init__(self, *, input_size: int, output_size: int):
    """
    model.jl:
    Npred_out = mp.Nout - Naction - 1
    prediction = Chain(Dense(mp.Nhidden+Naction, Npred_out, relu), Dense(Npred_out, Npred_out))
    """
    super().__init__()
    assert input_size > 0 and output_size > 0
    self.input_size = input_size
    self.output_size = output_size
    self.linear0 = nn.Linear(input_size, output_size)
    self.linear1 = nn.Linear(output_size, output_size)
  
  def forward(self, y: torch.Tensor):
    """
    Note that we do not have a final nonlinearity here because the output is (may be) interpreted 
    elsewhere as a distribution over states
    """
    return self.linear1(torch.relu(self.linear0(y))) 
  
  def ctor_params(self):
    return {'input_size': self.input_size, 'output_size': self.output_size}

class AgentModel(nn.Module):
  def __init__(self, *, rnn: RNN, policy: Policy, prediction: Prediction):
    super().__init__()
    self.rnn = rnn
    self.policy = policy
    self.prediction = prediction

  def ctor_params(self):
    return {
      'rnn': self.rnn.ctor_params(),
      'policy': self.policy.ctor_params(), 
      'prediction': self.prediction.ctor_params()
    }
  
  @staticmethod
  def from_ctor_params(params: Dict):
    rnn = RNN(**params['rnn'])
    policy = Policy(**params['policy'])
    prediction = Prediction(**params['prediction'])
    return AgentModel(rnn=rnn, policy=policy, prediction=prediction)