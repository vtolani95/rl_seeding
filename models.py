import numpy as np
import tensorflow as tf

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

