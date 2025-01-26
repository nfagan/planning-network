import eval
import env
import torch
import os

def main():
  prefer_gpu = False
  s = 4 # arena side length
  batch_size = 40
  num_episodes = 50000
  device = torch.device('cuda:0' if prefer_gpu and torch.cuda.is_available() else 'cpu')

  meta = eval.make_meta(arena_len=s, batch_size=batch_size, plan_len=8, device=device)
  model = eval.build_model(meta=meta)

  optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

  for e in range(num_episodes):
    print(f'{e+1} of {num_episodes}')
    mazes = env.build_maze_arenas(s, meta.batch_size)
    res = eval.run_episode(meta, model, mazes)
    loss = res.loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    if (e == num_episodes - 1) or e % int(5e3) == 0:
      save_p = os.path.join(os.getcwd(), 'checkpoints', f'cp-{e}.pth')
      torch.save({'state': model.state_dict()}, save_p)

if __name__ == '__main__':
  main()