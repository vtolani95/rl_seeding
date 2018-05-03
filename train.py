from envs_rl_seeding.envs import SocialNetworkGraphEnv
from trainers.dqn_trainer_v0 import DQNTrainer, get_dqn_v0_args
from trainers.random_trainer_v0 import RandomTrainer, get_random_v0_args
from trainers.exhaustive_trainer_v0 import ExhaustiveTrainer, get_exhaustive_v0_args
from src import utils
from _logging import logging
from baselines import logger
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import pdb


FLAGS = flags.FLAGS
flags.DEFINE_string('logdir_suffix', '', '')
flags.DEFINE_string('logdir_prefix', 'output/', '')
flags.DEFINE_string('config_name', '', '')

def get_other_args(str_):
  t = [('seed', '0'), ('social_network_graph_env', '0'),
        ('num_valid', '50')]
  da = utils.DefArgs(t)
  args = da.process_string(str_)
  vs = vars(args)
  for k in ['seed', 'num_valid']:
    vs[k] = int(utils.str_to_float(vs[k]))

  vs['social_network_graph_env'] = {'0':'SocialNetworkGraphEnv-v1'}[vs['social_network_graph_env']]
  return vs

def main(_):
  config = tf.ConfigProto();
  config.device_count['GPU'] = 1
  config.gpu_options.allow_growth = True
  config.intra_op_parallelism_threads = 1
  config.inter_op_parallelism_threads = 1

  config_name = FLAGS.config_name
  env_str, trainer_str, other_str = config_name.split('.')
  other_kwargs = get_other_args(other_str)

  env_kwargs = SocialNetworkGraphEnv.get_env_args(env_str)
  
  if env_kwargs['method_name'] == 'dqnV0':
    trainer_kwargs, trainer_name = get_dqn_v0_args(trainer_str), 'dqnV0'
  elif env_kwargs['method_name'] == 'randomV0':
    trainer_kwargs, trainer_name = get_random_v0_args(trainer_str), 'randomV0'
  elif env_kwargs['method_name'] == 'exhaustiveV0':
    trainer_kwargs, trainer_name = get_exhaustive_v0_args(trainer_str), 'exhaustiveV0'
  else:
    assert(False)

  logdir = FLAGS.logdir_prefix + FLAGS.config_name + FLAGS.logdir_suffix
  logger.configure(logdir)
  logger.error('env_kwargs: ', env_kwargs)
  logger.error('other_kwargs: ', other_kwargs)
  logger.error('%s_kwargs: '%(trainer_name), trainer_kwargs)

  tf.set_random_seed(other_kwargs['seed'])
  random.seed(other_kwargs['seed'])
  np.random.seed(other_kwargs['seed'])

  if env_kwargs['method_name'] == 'dqnV0':
    dqnTrainer = DQNTrainer()
    dqnTrainer.train()
  elif env_kwargs['method_name'] == 'randomV0':
    randomTrainer = RandomTrainer(env_name=other_kwargs['social_network_graph_env'],
                                  env_kwargs=env_kwargs,
                                  trainer_kwargs=trainer_kwargs,
                                  other_kwargs=other_kwargs,
                                  logdir=logdir)
    randomTrainer.train()
  elif env_kwargs['method_name'] == 'exhaustiveV0':
    exhaustiveTrainer = ExhaustiveTrainer()
    exhaustiveTrainer.train()
  else:
    assert(False)


if __name__ == '__main__':
  app.run()
