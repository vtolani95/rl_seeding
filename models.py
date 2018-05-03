import numpy as np
import tensorflow as tf

def _random_policy(obs, num_actions):
  return np.random.choice(num_actions) 
