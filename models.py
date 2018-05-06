import numpy as np
import tensorflow as tf
import itertools

def _random_policy(obs, env_val):
  return np.random.choice(env_val.num_actions) 

#seed the B supporter node with the most neighbors
#that are B supporters
def _greedy_policy(obs, env_val):
  pro_A = env_val.pro_A*1.
  G = env_val.G
  opportunities = []
  for node in G.nodes:
    count = 0
    if pro_A[node] == 0.0:#this node is a B supporter
      for neighbor in G.neighbors(node):
        if pro_A[neighbor] == 0.0:#B supporter
          count += 1
    opportunities.append(count)
  return np.argmax(opportunities) 

#Do an exhaustive search over all permutations
#of seeding k B supporters, and pick the best one
def _exhaustive_policy(obs, env_val):
  pro_A = env_val.pro_A*1.
  pro_B_idx = list(np.argwhere(pro_A==0)[:,0])
  print('This is impractically slow. For a graph with 64 nodes where 54 are initially B supporters and we look for\
  all the possible permutations of size 10 there are 8.6e16 possibilities')
  perms = list(itertools.permutations(pro_B_idx, env_val.T))
  
