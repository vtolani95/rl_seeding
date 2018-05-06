import gym
import matplotlib.pyplot as plt
from src import utils
import numpy as np
import os
import tensorflow as tf
from models import _dqn_policy
from src.tf_utils import add_value_to_summary
from baselines import deepq, logger


def get_dqn_v0_args(str_):
  t = [('lr', '5en4'), ('max_timesteps', '1e7'), ('buffer_size', '5e4'),
    ('exploration_fraction', '1en1'), ('exploration_final_eps', '0x02'),
    ('train_freq', '100.'), ('batch_size', '32'), ('learning_starts', '5000'),
    ('gamma', '1x0'), ('target_network_update_freq', '1e5'), 
    ('prioritized_replay', '0'), ('cnn_model', '0'), ('print_freq', '1000')]
  da = utils.DefArgs(t)
  args = da.process_string(str_)
  vs = vars(args)
  for k in ['batch_size', 'buffer_size', 'learning_starts', 'max_timesteps',
      'target_network_update_freq', 'prioritized_replay', 'cnn_model', 'print_freq']:
    vs[k] = int(utils.str_to_float(vs[k]))

  vs['prioritized_replay'] = vs['prioritized_replay'] > 0
  
  for k in ['lr', 'exploration_fraction', 'exploration_final_eps', 'gamma', 'train_freq']:
    vs[k] = utils.str_to_float(vs[k])
  return vs

class DQNTrainer():
  def __init__(self, env_name, env_kwargs, trainer_kwargs, other_kwargs, logdir):
    self.env_name = env_name
    self.env_kwargs = env_kwargs
    self.trainer_kwargs = trainer_kwargs
    self.other_kwargs = other_kwargs
    self.env_train = gym.make(env_name)
    self.env_val = gym.make(env_name)

    if type(env_kwargs) == list:
      self.env_train.configure(env_kwargs)
      self.env_val.configure(env_kwargs)
    else:
      self.env_train.configure(purpose='training', **env_kwargs)
      self.env_val.configure(purpose='test', **env_kwargs)

    self.pi = lambda *x, **y: _dqn_policy([512, 512, 512], cnn_model=self.trainer_kwargs['cnn_model'], env=self.env_train, *x, **y) 
  
    self.logdir = logdir
    utils.mkdir_if_missing(self.logdir)
    utils.mkdir_if_missing(os.path.join(logdir, 'val'))
    self.logging = utils.Foo()
    self.logging.writer_val = \
      tf.summary.FileWriter(os.path.join(logdir, 'val'), flush_secs=20)
    self.iter = 0

  def train(self, config):
    act = deepq.learn(self.env_train, q_func=self.pi,
      tf_session_config=config, callback=self.callback, checkpoint_freq=None,
      **self.trainer_kwargs)
  
  def callback(self, lcl, glb):
    self.iter += 1
    if self.iter == 1:
      self.sess = lcl['sess']
      self.callback_setup_saver()
      self.logging.writer_val.add_graph(lcl['sess'].graph)
      self.callback_val_vis(lcl, glb,
                          num_rollouts=self.other_kwargs['num_valid'],
                          plot=True)
      return
    t = lcl['t']
    print_freq = lcl['print_freq']
    if t > self.trainer_kwargs['learning_starts']:
      if t % print_freq == 0:
        self.callback_val_vis(lcl, glb,
                          num_rollouts=self.other_kwargs['num_valid'],
                          plot=True)
      if  t % (print_freq*15) == 0:
        self.callback_snapshot(lcl, glb) 
      
  def callback_setup_saver(self):
    self.logging.saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, 
      max_to_keep=8, pad_step_number=True)

  def callback_snapshot(self, lcl, glb):
    model_file_name = os.path.join(self.logdir, 'snapshots', 'model')
    self.logging.saver.save(lcl['sess'], model_file_name, global_step=lcl['num_episodes'])
    logger.error('Saving model to: ', model_file_name)  

  def callback_val_vis(self, lcl, glb, num_rollouts, plot=False):
    act = lcl['act']
    global_step = lcl['t']
    with plt.style.context("fivethirtyeight"):
      plt.rcParams["axes.grid"] = True
      env = self.env_val
      env.reset_rng()
      obsss, actionss, rewardss = [], [], []
      ms = []
      for i in range(num_rollouts):
        obs, done = env.reset(), False
        obss, actions, rewards = [obs], [], []
        while not done:
          action = act(obs[None], False)[0]
          obs, rew, done, _ = env.step(action)
          obss.append(obs); actions.append(action); rewards.append(rew)
        obss.pop()#last obs is unnecessary
        obsss.append(obss); actionss.append(actions); rewardss.append(rewards)  
        m = env.get_metrics()
        ms.append(m)
      
      metric_names, metric_vals = env.collect_metrics(ms)
      metric_summary_init = tf.summary.Summary()
      metric_summary_end = tf.summary.Summary()
      for k, v in zip(metric_names, metric_vals):
        add_value_to_summary(metric_summary_init,
                             'metrics/{:s}'.format(k), v, log=True, tag_str='metrics/{:s}: '.format(k))
        add_value_to_summary(metric_summary_end,
                             'metrics/{:s}'.format(k), v, log=False, tag_str='metrics/{:s}: '.format(k))
      
      self.logging.writer_val.add_summary(metric_summary_init, global_step)

