# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


#import atari_py
import minatar
import numpy as np
import tensorflow as tf

import gin.tf
#import cv2

slim = tf.contrib.slim


NATURE_DQN_OBSERVATION_SHAPE = (10, 10, 4)  # Size of MinAtar frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 1  # Number of frames in the state stack.

gin.constant('minatar_env.ASTERIX_SHAPE', (10, 10, 4)) 
gin.constant('minatar_env.BREAKOUT_SHAPE', (10, 10, 4)) 
gin.constant('minatar_env.FREEWAY_SHAPE', (10, 10, 7)) 
gin.constant('minatar_env.SEAQUEST_SHAPE', (10, 10, 10))
gin.constant('minatar_env.SPACE_INVADERS_SHAPE', (10, 10, 6)) 
#gin.constant('minatar_env.DTYPE', jnp.float64)



@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True):
  print ('STICKY_ACTIONS: (still not being used)', sticky_actions)
  return MinAtarEnv(game_name)


def nature_dqn_network(num_actions, network_type, state, aux=False, next_state=None):
  """The convolutional network used to compute the agent's Q-values.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  state = tf.squeeze(state, 4)
  net = tf.cast(state, tf.float32)
  # net = tf.div(net, 255.)
  net = slim.conv2d(net, 16, [3, 3], stride=1, scope='conv2d_1')
  net = slim.flatten(net)

  net = slim.fully_connected(net, 128)
  q_values = slim.fully_connected(net, num_actions, activation_fn=None)
  return network_type(q_values, None)

#@profile
def rainbow_network(num_actions, num_atoms, num_atoms_sub, support, network_type, state, runtype='run', v_support=None, a_support=None, big_z=None, big_a=None, big_qv=None, N=1, index=None, M=None, sp_a=None, unique_num=None, sortsp_a=None, v_sup_tensor=None): #run, conv, convmean
  """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  state = tf.squeeze(state, 4)
  net = tf.cast(state, tf.float32)
  # net = tf.div(net, 255.)
  state_input = net
  net = slim.conv2d(
      net, 16, [3, 3], stride=1, weights_initializer=weights_initializer)
  feature = slim.flatten(net)
  feature_size = int(feature.shape[-1])
  a_origin, Ea = None, None
  net = slim.fully_connected(feature, 128, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)
  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities, None, None, None, a_origin, Ea, None, None, None)

#@profile
def implicit_quantile_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles):
  """The Implicit Quantile ConvNet.

  Args:
    num_actions: int, number of actions.
    quantile_embedding_dim: int, embedding dimension for the quantile input.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    num_quantiles: int, number of quantile inputs.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  state = tf.squeeze(state, 4)
  state_net = tf.cast(state, tf.float32)
  # state_net = tf.div(state_net, 255.)
  state_net = slim.conv2d(
      state_net, 16, [3, 3], stride=1,
      weights_initializer=weights_initializer)
  state_net = slim.flatten(state_net)
  print ('state_net:', state_net.shape, ", num_quan:", num_quantiles)
  state_net_size = state_net.get_shape().as_list()[-1]

  state_net_tiled = tf.tile(state_net, [num_quantiles, 1])
  batch_size = state_net.get_shape().as_list()[0]
  quantiles_shape = [batch_size * num_quantiles, 1]
  quantiles = tf.random_uniform(
      quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

  quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
  quantile_net = tf.cos(quantile_net)
  quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                      weights_initializer=weights_initializer)
  net = tf.multiply(state_net_tiled, quantile_net)
  net = slim.fully_connected(
      net, 128, weights_initializer=weights_initializer)
  quantile_values = slim.fully_connected(
      net,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer)
  return network_type(quantile_values=quantile_values, quantiles=quantiles)

def fqf_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles, runtype='fqf'):
  """The FQF ConvNet.

  Args:
    num_actions: int, number of actions.
    quantile_embedding_dim: int, embedding dimension for the quantile input.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    num_quantiles: int, number of quantile inputs.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)


  state = tf.squeeze(state, 4)
  state_net = tf.cast(state, tf.float32)
  # state_net = tf.div(state_net, 255.)
  state_net = slim.conv2d(
      state_net, 16, [3, 3], stride=1,
      weights_initializer=weights_initializer)
  state_net = slim.flatten(state_net)
  print ('state_net:', state_net.shape, ", num_quan:", num_quantiles)
  state_net_size = state_net.get_shape().as_list()[-1]

  quantile_values_origin = None
  quantiles_origin = None
  Fv_diff = None
  v_diff = None
  L_tau = None
  quantile_values_mid = None
  quantiles_mid = None
  gradient_tau = None
  quantile_tau = None

  batch_size = state_net.get_shape().as_list()[0]
  state_net1 = state_net

  quantiles_right = slim.fully_connected(state_net1, num_quantiles, weights_initializer=weights_initializer, scope='fqf', reuse=False, activation_fn=None)
  quantiles_right = tf.reshape(quantiles_right, [batch_size, num_quantiles])
  quantiles_right = tf.contrib.layers.softmax(quantiles_right) * (1 - 0.00)
  zeros = tf.zeros([batch_size, 1])
  quantiles_right = tf.cumsum(quantiles_right, axis=1)  #batchsize x 32
  quantiles_all = tf.concat([zeros, quantiles_right], axis=-1)   #33
  quantiles_left = quantiles_all[:, :-1]  #32
  quantiles_center = quantiles_all[:, 1:-1] #31, delete 0&1
  quantiles = quantiles_center
  quantiles_mid = (quantiles_right + quantiles_left) / 2  #batchsize x 32
  v_diff = quantiles_right - quantiles_left  #32
  v_diff = tf.transpose(v_diff, [1, 0])  #quan x batchsize

  quantile_tau = quantiles
  quantile_tau = tf.transpose(quantile_tau, [1, 0])  #quan x batchsize
  quantile_tau = tf.reshape(quantile_tau, [num_quantiles-1, batch_size])

  quantiles = tf.transpose(quantiles, [1, 0])  #quan-1 x batchsize
  quantiles = tf.reshape(quantiles, [(num_quantiles-1) * batch_size , 1])
  quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
  quantile_net = tf.cos(quantile_net)
  quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                      weights_initializer=weights_initializer, scope='quantile_net')

  quantiles_mid = tf.transpose(quantiles_mid, [1, 0])  #quan x batchsize
  quantiles_mid = tf.reshape(quantiles_mid, [(num_quantiles)*batch_size, 1])
  quantile_net_mid = tf.tile(quantiles_mid, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net_mid = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net_mid
  quantile_net_mid = tf.cos(quantile_net_mid)
  quantile_net_mid = slim.fully_connected(quantile_net_mid, state_net_size,
                                      weights_initializer=weights_initializer, scope='quantile_net', reuse=True)
  # Hadamard product.
  state_net_tiled = tf.tile(state_net, [num_quantiles - 1, 1])
  net = tf.multiply(state_net_tiled, quantile_net)
  net = slim.fully_connected(
      net, 128, weights_initializer=weights_initializer)
  quantile_values = slim.fully_connected(
      net,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer,
      scope='quantile_values_net')

  state_net_tiled1 = tf.tile(state_net, [num_quantiles, 1])
  net1 = tf.multiply(state_net_tiled1, quantile_net_mid)
  net1 = slim.fully_connected(
      net1, 128, weights_initializer=weights_initializer)
  quantile_values_mid = slim.fully_connected(
      net1,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer,
      scope='quantile_values_net', reuse=True)

  quantile_values = tf.reshape(quantile_values, [num_quantiles-1, batch_size, num_actions])
  quantile_values_mid = tf.reshape(quantile_values_mid, [num_quantiles, batch_size, num_actions])
  quantile_values_mid_1 = quantile_values_mid[:-1, :, :]
  quantile_values_mid_2 = quantile_values_mid[1:, :, :]
  sum_1 = 2 * quantile_values  #31
  sum_2 = quantile_values_mid_2 + quantile_values_mid_1  #31
  L_tau = tf.square(sum_1 - sum_2)  #31 x batchsize x action
  gradient_tau = sum_1 - sum_2
  print ("sum_1:", sum_1.shape)
  print ("L_tau:", L_tau.shape)
  quantile_values_mid = tf.reshape(quantile_values_mid, [-1, num_actions])  #32 x batchsize x action
  quantile_values_mid = tf.reshape(quantile_values_mid, [-1, num_actions])  #32 x batchsize x action

  quantiles_mid = tf.reshape(quantiles_mid, [-1, 1])  #32 x batchsize x action
  #quantile_values = quantile_values_mid
  #quantile = quantile_mid
  return network_type(quantile_values=quantile_values_mid, quantiles=quantiles_mid, quantile_values_origin=quantile_values_origin, quantiles_origin=quantiles_origin, Fv_diff=Fv_diff, v_diff=v_diff, quantile_values_mid=quantile_values_mid, quantiles_mid=quantiles_mid, L_tau=L_tau, gradient_tau=gradient_tau, quantile_tau=quantile_tau)


@gin.configurable
class MinAtarEnv(object):
  def __init__(self, game_name):
    self.env = minatar.Environment(env_name=game_name)
    self.env.n = self.env.num_actions()
    self.game_over = False

  @property
  def observation_space(self):
    return self.env.state_shape()

  @property
  def action_space(self):
    return self.env  # Only used for the `n` parameter.

  @property
  def reward_range(self):
    pass  # Unused

  @property
  def metadata(self):
    pass  # Unused

  def reset(self):
    self.game_over = False
    self.env.reset()
    return self.env.state()

  def step(self, action):
    r, terminal = self.env.act(action)
    self.game_over = terminal
    return self.env.state(), r, terminal, None