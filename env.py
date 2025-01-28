import numpy as np
from typing import List

"""
0: right
1: left
2: up
3: down
dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
"""

_MOVES = [[1, 0], [-1, 0], [0, 1], [0, -1]]

def _actions():
  return np.arange(4)

def opposite_action(a: int):
  if a <= 1: return 1 - a
  return (1 - (a - 2)) + 2

def coord_to_index(x: int, y: int, s: int) -> int:
  return x * s + y

def index_to_coord(ind: int, s: int):
  x = ind // s
  y = ind - x * s
  return x, y

def _neighbors(i: int, s: int):
  xi, yi = index_to_coord(i, s)
  lx = (xi - 1) % s
  rx = (xi + 1) % s
  uy = (yi + 1) % s
  dy = (yi - 1) % s
  l = coord_to_index(lx, yi, s)
  r = coord_to_index(rx, yi, s)
  u = coord_to_index(xi, uy, s)
  d = coord_to_index(xi, dy, s)
  return [r, l, u, d], _actions()

class Arena(object):
  def __init__(self, s: int):
    """
    self.walls[i, j, x] is True if there is a wall between state i and state j via action x. Note 
    that `s` is the size of one side-length of the grid, the total number of states is s^2
    """
    # l = s * s
    l = s
    self.s = s
    self.walls = np.zeros((l, l, 4), dtype=bool)

  def clone(self):
    res = Arena(self.s)
    res.walls = self.walls.copy()
    return res

  def traversible(self, i: int, a: int):
    """
    True if taking action `a` in state `i` would not pass through a wall.
    """
    x, y = index_to_coord(i, self.s)
    return self.walls[y, x, a]

  def fill_walls(self):
    self.walls[:] = True

  def remove_wall(self, s0: int, s1: int, a: int):
    ah = opposite_action(a)
    ix0, iy0 = index_to_coord(s0, self.s)
    ix1, iy1 = index_to_coord(s1, self.s)
    self.walls[iy0, ix0, a] = False
    self.walls[iy1, ix1, ah] = False

  def neighbors(self, i: int):
    return _neighbors(i, self.s)
  
  def remove_random_wall(self):
    ij = np.argwhere(self.walls)
    assert ij.size > 0
    np.random.shuffle(ij)
    self.remove_wall(ij[0][0], ij[0][1], ij[0][2])

def build_fixed_maze_arenas(s: int, n: int) -> List[Arena]:
  if n <= 0:
    return []
  res = [build_maze_arena(s)]
  for _ in range(n - 1):
    res.append(res[0].clone())
  return res

def build_maze_arenas(s: int, n: int):
  """
  construct `n` traversible maze arenas with side length `s`
  """
  return [build_maze_arena(s) for _ in range(n)]

def build_maze_arena(s: int):
  """
  construct a traversible maze arena with side length `s`
  """
  l = s * s
  arena = Arena(s)
  arena.fill_walls()
  v = np.zeros((l,), dtype=bool)

  def walk(s):
    v[s] = True
    ns, acts = arena.neighbors(s)
    ind = np.arange(acts.size)
    np.random.shuffle(ind)
    for i in range(acts.size):
      j = ind[i]
      if not v[ns[j]]:
        arena.remove_wall(s, ns[j], acts[j])
        walk(ns[j])

  s0 = np.random.choice(l)
  walk(s0)

  for _ in range(3):
    arena.remove_random_wall()

  return arena

def move(i: int, a: int, s: int):
  xi, yi = index_to_coord(i, s)
  off = _MOVES[a]
  xj = (xi + off[0]) % s
  yj = (yi + off[1]) % s
  return coord_to_index(xj, yj, s)

def move_agent(i: int, a: int, arena: Arena):
  """
  move the agent from state `i` via action `a` in `arena`. The agent will remain in `i` if taking
  `a` would require it to move through a wall.
  """
  return move(i, a, arena.s) if arena.traversible(i, a) else i