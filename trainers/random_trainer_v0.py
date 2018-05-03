import gym
import matplotlib.pyplot as plt
from src import utils
import numpy as np
import os
import tensorflow as tf
from models import _random_policy
from src.tf_utils import add_value_to_summary

def get_random_v0_args(str_):
  t = [('max_episodes', '5e3')]
  da = utils.DefArgs(t)
  args = da.process_string(str_)
  vs = vars(args)
  for k in ['max_episodes']:
    vs[k] = int(utils.str_to_float(vs[k]))
  return vs

class RandomTrainer():
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

    self.pi = lambda obs: _random_policy(obs, self.env_val.num_actions) 
  
    self.logdir = logdir
    utils.mkdir_if_missing(self.logdir)
    utils.mkdir_if_missing(os.path.join(logdir, 'val'))
    self.logging = utils.Foo()
    self.logging.writer_val = \
      tf.summary.FileWriter(os.path.join(logdir, 'val'), flush_secs=20)

  def train(self, config=None):
    self.callback()

  def callback(self):
    self.callback_val_vis(global_step=self.trainer_kwargs['max_episodes'],
                          num_rollouts=self.other_kwargs['num_valid'],
                          plot=True)

  def callback_val_vis(self, global_step, num_rollouts, plot=False):
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
          action = self.pi(obs)
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
                             'metrics/{:s}'.format(k), v, log=True, tag_str='metrics/{:s}: '.format(k))
      
      self.logging.writer_val.add_summary(metric_summary_init, 0)
      self.logging.writer_val.add_summary(metric_summary_end, global_step)

