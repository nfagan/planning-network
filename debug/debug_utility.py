import sys; sys.path.append('..')
import model
import env
import torch

def put_rnn(net: model.RNN, cp: dict):
  net.gru.weight_ih_l0.data[:] = torch.tensor(cp['net_gru_wi']).type(torch.float32)
  net.gru.bias_ih_l0.data[:] = torch.tensor(cp['net_gru_b']).type(torch.float32).squeeze(1)

  net.gru.weight_hh_l0.data[:] = torch.tensor(cp['net_gru_wh']).type(torch.float32)
  net.gru.bias_hh_l0.data[:] = torch.tensor(cp['net_gru_b']).type(torch.float32).squeeze(1)
  
  if True: net.gru.bias_hh_l0.data.fill_(0.)
  if True: net.state0 = torch.tensor(cp['net_gru_state']).type(torch.float32).squeeze(1)

def put_pred(pred: model.Prediction, cp: dict):
  pred.linear0.weight.data[:] = torch.tensor(cp['pred_w1']).type(torch.float32)
  pred.linear0.bias.data[:] = torch.tensor(cp['pred_b1']).type(torch.float32).squeeze(1)

  pred.linear1.weight.data[:] = torch.tensor(cp['pred_w2']).type(torch.float32)
  pred.linear1.bias.data[:] = torch.tensor(cp['pred_b2']).type(torch.float32).squeeze(1)

def put_policy(policy: model.Policy, cp: dict):
  policy.linear.weight.data[:] = torch.tensor(cp['pol_w1']).type(torch.float32)
  policy.linear.bias.data[:] = torch.tensor(cp['pol_b1']).type(torch.float32).squeeze(1)

def put_params(model: model.AgentModel, kris_cp: dict):
  put_rnn(model.rnn, kris_cp)
  put_pred(model.prediction, kris_cp)
  put_policy(model.policy, kris_cp)

def arenas_from_kris_walls(kris_walls, s: int):
  fixed_mazes = []
  for bi in range(kris_walls.shape[2]):
    maze = env.Arena(s)
    for x in range(s):
      for y in range(s):
        si = env.coord_to_index(x, y, s)
        wc = kris_walls[si, :, bi]  
        maze.walls[y, x, :] = wc
    fixed_mazes.append(maze)
  return fixed_mazes