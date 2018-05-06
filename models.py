import numpy as np
import tensorflow as tf
import itertools
import tensorflow.contrib.layers as layers

def cnn(convs, out, reuse, pooling=False):
  outs = []
  with tf.variable_scope("cnn", reuse=reuse):
    for i, (num_outputs, kernel_size, stride) in enumerate(convs):
      out = layers.convolution2d(out, num_outputs=num_outputs,
        kernel_size=kernel_size, stride=stride, activation_fn=tf.nn.relu, 
        scope='conv{:d}'.format(i))
      outs.append(out)
      if pooling:
        out = layers.max_pool2d(out, kernel_size=3,
              stride=2, padding='same', 
              scope='conv{:d}'.format(i))
      outs.append(out)
  return out, outs

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


def _dqn_policy(hiddens, inpt, num_actions, scope, env, cnn_model, reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    inputs = env._deserialize_obs(inpt)
    img, other = inputs['img'], inputs['other']
  
    with tf.variable_scope('other', reuse=reuse):
      out = layers.fully_connected(other, num_outputs=512, activation_fn=tf.nn.relu, scope='other_encoding')

    if cnn_model in [0]:
      _out, _outs = cnn([(64,3,2), (64,3,2), (64,3,2), (32,3,2)], img, reuse)
    else:
      assert(False)

    img_f = layers.flatten(_out, scope='flatten')
    out = tf.concat([out, img_f], axis=1)
    
    with tf.variable_scope('mlp', reuse=reuse):
      for hidden in hiddens:
        out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)

    with tf.variable_scope('out', reuse=reuse):
      q_out = layers.fully_connected(out, num_outputs=1, activation_fn=None)

    return q_out 
