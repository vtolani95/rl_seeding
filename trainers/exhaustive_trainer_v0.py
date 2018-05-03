import numpy as np
import tensorflow as tf

def get_exhaustive_v0_args(str_):
  t = [('lr', '5en4'), ('max_timesteps', '1e5'), ('buffer_size', '5e4'),
    ('exploration_fraction', '1en1'), ('exploration_final_eps', '0x02'),
    ('train_freq', '1.'), ('batch_size', '32'), ('learning_starts', '1000'),
    ('gamma', '1x0'), ('target_network_update_freq', '500'), 
        ('prioritized_replay', '0'), ('q_is_training', '0'),
        ('learning_schedule', '0'), ('max_episodes', '0'), ('initial_exploration_p', '1x0'),
        ('num_update_steps', '1')]
  da = utils.DefArgs(t)
  args = da.process_string(str_)
  vs = vars(args)
  for k in ['batch_size', 'buffer_size', 'learning_starts', 'max_timesteps',
      'target_network_update_freq', 'prioritized_replay', 'q_is_training', 'learning_schedule', 'max_episodes', 'num_update_steps']:
    vs[k] = int(utils.str_to_float(vs[k]))

  vs['prioritized_replay'] = vs['prioritized_replay'] > 0
  
  for k in ['lr', 'exploration_fraction', 'exploration_final_eps', 'gamma', 'train_freq', 'initial_exploration_p']:
    vs[k] = utils.str_to_float(vs[k])
  vs['train_freq'] = num_waypts*vs['train_freq'] #train @train_freq/episode
  return vs

class ExhaustiveTrainer():
  def __init__():
    x = 0
