import gym
from gym import spaces
from gym.utils import seeding
from src import utils
import numpy as np
import networkx as nx
import tensorflow as tf

class SocialNetworkGraphEnv(gym.Env):

  @staticmethod
  def get_env_args(str_):
    t = [('method_name', 'ours'), ('graph_n', '64'),
        ('edges_n', '200'), ('n_init_a', '10'), 
        ('threshold_q', '.5'), ('num_seeds', '4'),
        ('reward_version', '0'), ('seed', '0'),
        ('graph_obs_version', '0')]
    da = utils.DefArgs(t)
    args = da.process_string(str_)
    vs = vars(args)

    for k in ['graph_n', 'edges_n', 'n_init_a', 'num_seeds', 'reward_version', 'seed', 'graph_obs_version']:
      vs[k] = int(utils.str_to_float(vs[k]))

    for k in ['threshold_q']:
      vs[k] = utils.str_to_float(vs[k])

    assert(vs['threshold_q'] >= 0 and vs['threshold_q'] <= 1) #valid percentage
    return vs

  @staticmethod
  def collect_metrics(ms):
    ms = np.array(ms)
    total_reward, final_pro_A, full_cascades, episode_len = ms.T
    keys = ['reward', 'percent_pro_A', 'full_cascades', 'episode_len']
    vals = [total_reward, final_pro_A, full_cascades, episode_len]
    fns = [np.mean, lambda x: np.percentile(x, q=25), lambda x:
      np.percentile(x, q=50), lambda x: np.percentile(x, q=75)]
    fn_names = ['mu', '25', '50', '75']
    out_vals, out_keys = [], []
    for k, v in zip(keys, vals):
      for fn, name in zip(fns, fn_names):
        _ = fn(v)
        out_keys.append('{:s}_{:s}'.format(k, name))
        out_vals.append(_)
    return out_keys, out_vals

  def get_metrics(self):
    total_reward = np.sum(self.rewards)
    final_pro_A = np.mean(self.pro_A)
    full_cascades = int(np.sum(self.pro_A) == len(self.pro_A)) 
    episode_len = self.t*1.
    tt = np.array([total_reward, final_pro_A, full_cascades, episode_len])
    return tt

  def configure(self, purpose, method_name, graph_n, edges_n,
                n_init_a, threshold_q, num_seeds,
                reward_version, seed, graph_obs_version):
    self.purpose = purpose
    self.T = num_seeds#number of people the agent can seed (max horizon for an episode)
    self.seed = seed
    self.rng = np.random.RandomState(self.seed)
    self.params = utils.Foo(method_name=method_name, graph_n=graph_n,
                            edges_n=edges_n, threshold_q=threshold_q,
                            reward_version=reward_version, n_init_a=n_init_a,
                            graph_obs_version=graph_obs_version)
    
    #setup observation and action spaces
    self.obs_sizes = (graph_n**2, graph_n, 2)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(np.sum(self.obs_sizes),))
    self.action_space = spaces.Discrete(graph_n)
    self.num_actions = graph_n
  
  @property
  def _seed(self):
    return self.seed

  def reset_rng(self):
    self.rng = np.random.RandomState(self.seed)

  def _reward(self):
    version = self.params.reward_version
    done = False
    if version == 0:#negative proportion of non A supporting nodes
      rew = -(1.0 - np.mean(self.pro_A))
      return rew, rew == 0.0
    else:
      assert(False) 

  def _step(self, u):
    self.t += 1
    pro_A = self.pro_A*1.
    pro_A[u] = 1.0
    for node in self.G.nodes:
      counts = []
      for neighbor in self.G.neighbors(node):
        if pro_A[neighbor] == 1.0:#A supporter
          counts.append(1.0)
        else:
          counts.append(0.0)#B supporter
      p = np.mean(counts)
      if p > self.params.threshold_q:
        pro_A[node] = 1.0
    
    #update environment and calculate reward  
    self.pro_A = pro_A 
    rew, done = self._reward()
    if self.t >= self.T:
      done = True

    #metrics
    self.percent_nodes_seeded.append(np.mean(pro_A))
    self.rewards.append(rew)

    return self._get_obs(), rew, done, {}

  def _reset(self):
    self.t = 0
    self._init_graph()
    #metrics
    self.percent_nodes_seeded = []
    self.rewards = []
    return self._get_obs()

  def _init_graph(self):
    rng, ng, ne, nA = self.rng, self.params.graph_n, self.params.edges_n, self.params.n_init_a
    G = nx.Graph()
    nodes = np.r_[:ng]
    G.add_nodes_from(nodes)
    for i in range(ne):#add random edges
      while True:
        n1, n2 = rng.choice(ng, 2)
        try:
          G[n1][n2]
          continue
        except KeyError: #edge not in graph, so add it
          G.add_edge(n1, n2)
          break
  
    pro_A = np.zeros(ng)#add random supporters of product A
    pro_A[rng.choice(ng, nA)] = 1.0
    self.G, self.pro_A = G, pro_A

  #turn the graph into a ngxng image
  #here a 1 represent an edge between nodes
  #and 0 represent no edge
  def _get_graph_obs(self):
    if self.params.graph_obs_version == 0:
      ng = self.params.graph_n
      img = np.eye(ng)
      for edge in self.G.edges():
        x,y = edge#implicitly undirected so mark both entries in the img
        img[x,y] = 1.0
        img[y,x] = 1.0
      return img
    else:
      assert(False)

  def _get_obs(self):
    t_obs = np.array([self.t*1., self.T*1.])
    obss = (self._get_graph_obs(), self.pro_A, t_obs)
    return self._serialize_obs(obss)

  def _serialize_obs(self, obss):
    t = [o.ravel().astype(np.float32) for o in obss]
    t = np.concatenate(t)
    return t

  def _deserialize_obs(self, obs):
    if self.params.graph_obs_version == 0:
      graph_size, pro_A_size, time_size = self.obs_sizes
      other_size = pro_A_size + time_size
      img, other = tf.split(obs, [graph_size, other_size], axis=1)
      img = tf.reshape(img, [-1, self.params.graph_n, self.params.graph_n, 1])
      other = tf.reshape(other, [-1, other_size])
      return {'img': img, 'other': other}
    else:
      assert(False) 
