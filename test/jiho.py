# #%%
import numpy as np

# num_games = 20

# numbers = np.arange(1, 46)
# prblty = 3 * np.ones_like(numbers).astype(float)
# weights = [[1, 3, 5, 6, 8, 11, 14, 16, 23, 33, 34, 45],
#           [7, 6, 5, 5, 6,  7,  7,  7,  6,  5,  6,  8]]
# for n, w in zip(*weights):
#     prblty[n-1] = w
# prblty = prblty / prblty.sum()

# for trial in range(num_games):
#     rots = np.random.choice(numbers, 6, False, p=prblty)
#     print(f"Game {trial}: ", end='')
#     for rot in np.sort(rots): print(f"{rot:02d}", end=' ')
#     print('')

# %%
from multiprocessing import Process, Manager, Pool
from functools import partial
import itertools

num = int(1E+4)

chunksize = 20 #this may take some guessing ... take a look at the docs to decide


def FF(cnts, lst):
    _n = np.random.randint(1, 46, 1)[0]
    cnts[_n] += 1
    print(f'iter = {lst}', end='\r')

pool = Pool()
manager = Manager()
counts = manager.dict()
for rot in range(1, 46):
    counts[rot] = 0
x = range(1, 46)
pool.map(func, [x])
pool.close()
pool.join()

idx = np.argsort(counts.values())[-6:]
print(counts.items())
