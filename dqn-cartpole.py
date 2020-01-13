import numpy as np
from math import ceil
import gym
from gym import wrappers
from sklearn.preprocessing import StandardScaler
import keras


'''
Iy has been shown thata an approximation of TD-Lambda is just Q-Learning with elegibility Trace. 
That means add momentum to gradients during training.
'''


discount = 0.99
eps = 0.1
hidden_layer = 100


class InputPreprocesser:
  def __init__(self):
    self.preprocesser = StandardScaler()

  def fit(self, env, size=20000):
    samples = [env.observation_space.sample() for _ in range(size)]
    self.preprocesser.fit(samples)

  def preprocess(self, x):
    return self.preprocesser.transform([x])


def get_deep_q_network(num_actions, hidden_layer, env_shape):
  input_ = keras.layers.Input(shape=(env_shape))
  h_out = keras.layers.Dense(hidden_layer, activation='relu')(input_)
  h_out2 = keras.layers.Dense(hidden_layer, activation='relu')(h_out)
  output = keras.layers.Dense(num_actions, activation='linear')(h_out2)
  model = keras.Model(input_, output)
  optimizer = keras.optimizers.Adam()
  model.compile(optimizer=optimizer, loss='mse')
  return model


class DQNAgent:
  def __init__(self, 
               num_actions, 
               hidden_layer, 
               env_shape, 
               update_freq,
               replay_max_size,
               batch_size):
    self.dqn = get_deep_q_network(num_actions, hidden_layer, env_shape)
    self.target_network = get_deep_q_network(num_actions, hidden_layer, env_shape)
    self.replay = []
    self.num_actions = num_actions
    self.updates = 0
    self.update_freq = update_freq
    self.replay_max_size = replay_max_size
    self.batch_size = batch_size
    self.env_shape = env_shape

  def update_agent(self, discount=0.9):
    if (self.updates + 1) % self.update_freq == 0:
      self.copy_dqn_to_target()
    
    self.updates += 1

    self.replay = self.replay[:self.replay_max_size]
    sars_samples_indexes = np.random.choice(len(self.replay), size=self.batch_size, replace=False)
    sars_samples = np.array(self.replay)[sars_samples_indexes]

    X, Y = [], []
    for s, a, r, s_, done in sars_samples:
      X.append(s)
      next_est = 0
      if not done:
        next_est = np.max(self.target_network.predict(np.array([s_])), axis=1)
      y = self.dqn.predict(np.array([s]))
      y[0][a] = r + discount * next_est
      Y.append(y)

    X = np.array(X).reshape((self.batch_size, self.env_shape[0]))
    Y = np.array(Y).reshape((self.batch_size, self.num_actions))
    self.dqn.train_on_batch(X, Y)

  def copy_dqn_to_target(self):
    self.dqn.save_weights('cartpole-weights/weights')
    self.target_network.load_weights('cartpole-weights/weights')

  def next_action(self, x, env, eps=0):
    if eps > np.random.rand():
        return env.action_space.sample()
    return np.argmax(self.dqn.predict(np.array([x]))[0])


def deep_q_learning(env,
                    agent, 
                    preprocesser,
                    train_freq = 100,
                    warmup = 50,
                    episodes=1000,
                    episode_max_len=1000):

  avg_reward = []
  
  for n in range(episodes):
    if (n+1) % 100 == 0:
      print('Starting episode: ', n+1, 'avg reward:', np.mean(avg_reward))
      avg_reward = []

    done = False
    count = 0

    s = env.reset()
    totalReward = 0

    while not done and count < episode_max_len:
      count += 1
      a = agent.next_action(s, env, eps=eps)
      s_, r, done, _ = env.step(a)

      if done and count < episode_max_len:
        r = -200
      totalReward += r
      agent.replay.insert(0, (s, a, r, s_, done))
      s = s_

      if n > warmup and (count + 1) % train_freq == 0:
        agent.update_agent(discount=discount)

    if n > warmup:
      agent.update_agent(discount=discount)

    avg_reward.append(totalReward)


def playEpisode(env, agent, preprocesser, max_len=10000):
  done = False
  s = env.reset()
  totalReward = 0
  count = 0
  while not done and count < max_len:
    count += 1
    action = agent.next_action(s, env)
    s, r, done, _ = env.step(action)
    totalReward += r
  
  return totalReward


if __name__ == '__main__':
  env = gym.make('CartPole-v0').env

  agent = DQNAgent(env.action_space.n, 
                  hidden_layer, 
                  env.observation_space.sample().shape,
                  100, #copy target iterations
                  10000, # max mem size
                  32)  # batch size
  preprocesser = InputPreprocesser()
  preprocesser.fit(env)

  deep_q_learning(env, agent, preprocesser, episodes=2000)

  # Show learned policy
  env = wrappers.Monitor(env, './video/dqn-cartpole')
  playEpisode(env, agent, preprocesser)