import torch
import torch.nn as nn

class RNN(nn.Module):
  def __init__(self, *, in_size: int, hidden_size: int):
    """
    model.jl:
    network = Chain(GRU(mp.Nin, mp.Nhidden))
    """
    super().__init__()
    assert in_size > 0 and hidden_size > 0
    self.hidden_size = hidden_size
    self.gru = nn.GRU(in_size, hidden_size, batch_first=True)

  def make_h0(self, batch_size: int, device: torch.device):
    return torch.zeros((batch_size, self.hidden_size), device=device)

  def forward(self, x: torch.tensor, h: torch.tensor):
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
    return y.squeeze(), h1.squeeze()

class Policy(nn.Module):
  def __init__(self, *, rnn_hidden_size: int, output_size: int):
    """
    model.jl:
    policy = Chain(Dense(mp.Nhidden, Naction+1)) #policy and value function
    """
    super().__init__()
    assert rnn_hidden_size > 0 and output_size > 0
    self.linear = nn.Linear(rnn_hidden_size, output_size)

  def forward(self, y: torch.tensor):
    """
    Note that we do not have a nonlinearity here because the output is interpreted to contain both a 
    policy and value function. In Kris' implementation, the softmax is computed later for the subset
    of components representing the policy; this is in the function `forward_modular`
    """
    return self.linear(y)
  
class Prediction(nn.Module):
  def __init__(self, *, input_size: int, output_size: int):
    """
    model.jl:
    Npred_out = mp.Nout - Naction - 1
    prediction = Chain(Dense(mp.Nhidden+Naction, Npred_out, relu), Dense(Npred_out, Npred_out))
    """
    super().__init__()
    assert input_size > 0 and output_size > 0
    self.linear0 = nn.Linear(input_size, output_size)
    self.linear1 = nn.Linear(output_size, output_size)
  
  def forward(self, y: torch.tensor):
    """
    Note that we do not have a final nonlinearity here because the output is (may be) interpreted 
    elsewhere as a distribution over states
    """
    return self.linear1(torch.relu(self.linear0(y))) 

class AgentModel(object):
  def __init__(self, *, rnn: RNN, policy: Policy, prediction: Prediction):
    self.rnn = rnn
    self.policy = policy
    self.prediction = prediction