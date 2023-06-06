import numpy as np
import math
import subprocess as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import tensorflow.nn as tfnn
import pickle
import random
from collections import deque



def initialize_fully_connected_layers(input_channels, dtype=tf.float32):
	def _initializer(shape, dtype=dtype, partition_info=None):
		d = 1.0 / np.sqrt(input_channels)
		return tf.random.uniform(shape, minval=-d, maxval=d)
	return _initializer


def initialize_convolutions(kernel_width, kernel_height, input_channels, dtype=tf.float32):
	def _initializer(shape, dtype=dtype, partition_info=None):
		d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
		return tf.random.uniform(shape, minval=-d, maxval=d)
	return _initializer


class Model(object):
	def __init__(self, action_size, thread_index, use_pixel_change, use_value_replay, use_reward_prediction, pixel_change_lambda, entropy_beta, device, for_display=False):

		self.device = device
		self.action_size = action_size
		self.thread_index = thread_index
		self.use_pixel_change = use_pixel_change
		self.use_value_replay = use_value_replay
		self.use_reward_prediction = use_reward_prediction
		self.pixel_change_lambda = pixel_change_lambda 
		self.entropy_beta = entropy_beta
		self.main_network(for_display)


	def main_network(self, for_display):
		scope_name = "net_{0}".format(self.thread_index)
		with tf.device(self.device), tf.compat.v1.variable_scope(scope_name) as scope:
			self.lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)

			self.a3c_network()

			if self.use_pixel_change:
				self.pixel_change()
				if for_display:
					self.pixel_change_for_display()

			if self.use_reward_prediction:
				self.reward_prediction()

			self.reset_state()

			self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
			print('VARIABLES', self.variables)


	def a3c_network(self):
		self.base_input = tf.compat.v1.placeholder("float32", [None, 84, 84, 3])

		self.base_last_action_reward_input = tf.compat.v1.placeholder("float32", [None, self.action_size+1])

		base_conv_output = self.conv_layers(self.base_input)

		self.base_initial_lstm_state0 = tf.compat.v1.placeholder("float32", [1, 256])
		self.base_initial_lstm_state1 = tf.compat.v1.placeholder("float32", [1, 256])

		self.base_initial_lstm_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(	self.base_initial_lstm_state0, self.base_initial_lstm_state1)


		self.base_lstm_outputs, self.base_lstm_state = self.lstm_layers(base_conv_output, self.base_last_action_reward_input, self.base_initial_lstm_state)
		self.base_pi = self.policy_layer(self.base_lstm_outputs)
		self.base_v  = self.value_layer(self.base_lstm_outputs)



	def conv_layers(self, state_input, reuse=False):
		with tf.compat.v1.variable_scope("conv_layers", reuse=reuse) as scope:
			W_conv1, b_conv1 = self.convs([8, 8, 3, 16],  "base_conv1")
			W_conv2, b_conv2 = self.convs([4, 4, 16, 32], "base_conv2")
			#W_conv1 = tf.compat.v1.convert_to_tensor(W_conv1)
			#W_conv2 = tf.compat.v1.convert_to_tensor(W_conv2)
			#b_conv1 = tf.compat.v1.convert_to_tensor(b_conv1)
			#b_conv2 = tf.compat.v1.convert_to_tensor(b_conv2)



			h_conv1 = tfnn.relu(self.conv2d(state_input, W_conv1, 4) + b_conv1) 
			h_conv2 = tfnn.relu(self.conv2d(h_conv1,     W_conv2, 2) + b_conv2)
			return h_conv2


	def lstm_layers(self, conv_output, last_action_reward_input, initial_state_input, reuse=False):
		with tf.compat.v1.variable_scope("base_lstm", reuse=reuse) as scope:
			W_fc1, b_fc1 = self.fully_connected_layer([2592, 256], "base_fc1")

			conv_output_flat = tf.reshape(conv_output, [-1, 2592])
			conv_output_fc = tfnn.relu(tf.linalg.matmul(conv_output_flat, W_fc1) + b_fc1)
			step_size = tf.shape(conv_output_fc)[:1]
			lstm_input = tf.concat([conv_output_fc, last_action_reward_input], 1)
			lstm_input_reshaped = tf.reshape(lstm_input, [1, -1, 256+self.action_size+1])
			lstm_outputs, lstm_state = tf.compat.v1.nn.dynamic_rnn(self.lstm_cell, lstm_input_reshaped, initial_state = initial_state_input, sequence_length = step_size, time_major = False, scope = scope)

			lstm_outputs = tf.reshape(lstm_outputs, [-1,256])
			return lstm_outputs, lstm_state


	def policy_layer(self, lstm_outputs, reuse=False):
		with tf.compat.v1.variable_scope("policy", reuse=reuse) as scope:
			W_fc_p, b_fc_p = self.fully_connected_layer([256, self.action_size], "base_fc_p")
			
			base_pi = tfnn.softmax(tf.linalg.matmul(lstm_outputs, W_fc_p) + b_fc_p)
			return base_pi


	def value_layer(self, lstm_outputs, reuse=False):
		with tf.compat.v1.variable_scope("value", reuse=reuse) as scope:
			W_fc_v, b_fc_v = self.fully_connected_layer([256, 1], "base_fc_v")

			v_ = tf.linalg.matmul(lstm_outputs, W_fc_v) + b_fc_v
			base_v = tf.reshape( v_, [-1] )
			return base_v







	def pixel_change(self):
		self.pc_input = tf.compat.v1.placeholder("float32", [None, 84, 84, 3])
		self.pc_last_action_reward_input = tf.compat.v1.placeholder("float32", [None, self.action_size+1])

		pc_conv_output = self.conv_layers(self.pc_input, reuse=True)
		pc_initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)
		pc_lstm_outputs, _ = self.lstm_layers(pc_conv_output, self.pc_last_action_reward_input, pc_initial_lstm_state, reuse=True)

		self.pc_q, self.pc_q_max = self.deconvolution_pixel_change(pc_lstm_outputs)


	def pixel_change_for_display(self):
		self.pc_q_disp, self.pc_q_max_disp = self.deconvolution_pixel_change(self.base_lstm_outputs, reuse=True)


	def deconvolution_pixel_change(self, lstm_outputs, reuse=False):
		with tf.compat.v1.variable_scope("deconv_pixel_change", reuse=reuse) as scope:    
			W_pc_fc1, b_pc_fc1 = self.fully_connected_layer([256, 32*32*3], "pc_fc1")

			W_pc_deconv_v, b_pc_deconv_v = self.convs([4, 4, 1, 32],
			                                                 "pc_deconv_v", deconv=True)
			W_pc_deconv_a, b_pc_deconv_a = self.convs([4, 4, self.action_size, 32],
			                                                 "pc_deconv_a", deconv=True)
			m_pc_fc1 = tfnn.relu(tf.linalg.matmul(lstm_outputs, W_pc_fc1) + b_pc_fc1)
			m_pc_fc1_reshaped = tf.reshape(m_pc_fc1, [-1,9,9,32])
			m_pc_deconv_v = tfnn.relu(self.deconv2d(m_pc_fc1_reshaped, W_pc_deconv_v, 9, 9, 2) + b_pc_deconv_v)
			m_pc_deconv_a = tfnn.relu(self.deconv2d(m_pc_fc1_reshaped, W_pc_deconv_a, 9, 9, 2) + b_pc_deconv_a)
			m_pc_deconv_a_mean = tf.math.reduce_mean(m_pc_deconv_a, axis=3, keepdims=True)

			pc_q = m_pc_deconv_v + m_pc_deconv_a - m_pc_deconv_a_mean
			pc_q_max = tf.math.reduce_max(pc_q, axis=3, keepdims=False)

			return pc_q, pc_q_max


	def reward_prediction(self):
		self.reward_prediction_input = tf.compat.v1.placeholder("float32", [3, 84, 84, 3])
		reward_prediction_conv_output = self.conv_layers(self.reward_prediction_input, reuse=True)
		reward_prediction_conv_output_reshaped = tf.reshape(reward_prediction_conv_output, [1,9*9*32*3])

		with tf.compat.v1.variable_scope("reward_prediction") as scope:
			W_fc1, b_fc1 = self.fully_connected_layer([9*9*32*3, 3], "reward_prediction_fc1")

		self.reward_prediction_c = tfnn.softmax(tf.linalg.matmul(reward_prediction_conv_output_reshaped, W_fc1) + b_fc1)






	def a3c_loss(self):
		self.base_a = tf.compat.v1.placeholder("float32", [None, self.action_size])

		self.base_adv = tf.compat.v1.placeholder("float32", [None])

		log_pi = tf.math.log(tf.clip_by_value(self.base_pi, 1e-20, 1.0))

		entropy = -tf.math.reduce_sum(self.base_pi * log_pi, axis=1)

		policy_loss = -tf.math.reduce_sum( tf.math.reduce_sum(tf.math.multiply( log_pi, self.base_a ), axis=1 ) * self.base_adv + entropy * self.entropy_beta)

		self.base_r = tf.compat.v1.placeholder("float32", [None])
		value_loss = 0.5 * tfnn.l2_loss(self.base_r - self.base_v)

		base_loss = policy_loss + value_loss
		return base_loss


	def pixel_change_loss(self):
		self.pc_a = tf.compat.v1.placeholder("float32", [None, self.action_size])
		pc_a_reshaped = tf.reshape(self.pc_a, [-1, 1, 1, self.action_size])

		pc_qa = tf.math.multiply(self.pc_q, pc_a_reshaped)
		pc_qa = tf.math.reduce_sum(pc_qa, axis=3, keepdims=False)

		self.pc_r = tf.compat.v1.placeholder("float32", [None, 20, 20])

		pixel_change_loss = self.pixel_change_lambda * tfnn.l2_loss(self.pc_r - pc_qa)
		return pixel_change_loss



	def reward_prediction_loss(self):
		self.reward_prediction_c_target = tf.compat.v1.placeholder("float", [1,3])
		reward_prediction_c = tf.clip_by_value(self.reward_prediction_c, 1e-20, 1.0)
		reward_prediction_loss = -tf.math.reduce_sum(self.reward_prediction_c_target * tf.math.log(reward_prediction_c))
		return reward_prediction_loss


	def loss_sum(self):
		with tf.device(self.device):
			loss = self.a3c_loss()

		if self.use_pixel_change:
			pixel_change_loss = self.pixel_change_loss()
			loss = loss + pixel_change_loss


		if self.use_reward_prediction:
			reward_prediction_loss = self.reward_prediction_loss()
			loss = loss + reward_prediction_loss

		self.total_loss = loss








	def reset_state(self):
		self.base_lstm_state_out = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))


	def run_pv(self, sess, s_t, last_action_reward):
		pi_out, v_out, self.base_lstm_state_out = sess.run([self.base_pi, self.base_v, self.base_lstm_state], feed_dict = {self.base_input : [s_t], self.base_last_action_reward_input : [last_action_reward], self.base_initial_lstm_state0 : self.base_lstm_state_out[0], self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
		return (pi_out[0], v_out[0])


	def run_pixel_change_pv(self, sess, s_t, last_action_reward):
		# For display tool.
		pi_out, v_out, self.base_lstm_state_out, q_disp_out, q_max_disp_out = sess.run( [self.base_pi, self.base_v, self.base_lstm_state, self.pc_q_disp, self.pc_q_max_disp], feed_dict = {self.base_input : [s_t], self.base_last_action_reward_input : [last_action_reward], self.base_initial_lstm_state0 : self.base_lstm_state_out[0], self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
		return (pi_out[0], v_out[0], q_disp_out[0])


	def run_value(self, sess, s_t, last_action_reward):
		v_out, _ = sess.run( [self.base_v, self.base_lstm_state],
		                     feed_dict = {self.base_input : [s_t], self.base_last_action_reward_input : [last_action_reward], self.base_initial_lstm_state0 : self.base_lstm_state_out[0], self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
		return v_out[0]


	def run_pixel_change_q_max(self, sess, s_t, last_action_reward):
		q_max_out = sess.run( self.pc_q_max,
		                      feed_dict = {self.pc_input : [s_t],
		                                   self.pc_last_action_reward_input : [last_action_reward]} )
		return q_max_out[0]



	def run_reward_prediction(self, sess, s_t):
		reward_prediction_c_out = sess.run( self.reward_prediction_c,
		                    feed_dict = {self.reward_prediction_input : s_t} )
		return reward_prediction_c_out[0]


	def get_vars(self):
		return self.variables


	def sync_from(self, src_network, name=None):
		src_vars = src_network.get_vars()
		dst_vars = self.get_vars()

		sync_ops = []

		with tf.device(self._device):
			with tf.compat.v1.keras.backend.name_scope(name, "A3C_Model",[]) as name:
				for(src_var, dst_var) in zip(src_vars, dst_vars):
					sync_op = tf.compat.v1.assign(dst_var, src_var)
					sync_ops.append(sync_op)

				return tf.group(*sync_ops, name=name)




	def fully_connected_layer(self, weight_shape, name):
		weight_name = "W_{0}".format(name)
		bias_name = "b_{0}".format(name)

		input_channels  = weight_shape[0]
		output_channels = weight_shape[1]
		bias_shape = [output_channels]

		weight = tf.compat.v1.get_variable(weight_name, weight_shape, initializer=initialize_fully_connected_layers(input_channels))
		bias   = tf.compat.v1.get_variable(bias_name, bias_shape, initializer=initialize_fully_connected_layers(input_channels))
		return weight, bias


	def convs(self, weight_shape, name, deconv=False):
		weight_name = "W_{0}".format(name)
		bias_name = "b_{0}".format(name)

		w = weight_shape[0]
		h = weight_shape[1]
		if deconv:
		  input_channels  = weight_shape[3]
		  output_channels = weight_shape[2]
		else:
		  input_channels  = weight_shape[2]
		  output_channels = weight_shape[3]
		bias_shape = [output_channels]

		#weight = tf.compat.v1.get_variable(weight_name, weight_shape, initializer=initialize_convolutions(w, h, input_channels))
		#bias   = tf.compat.v1.get_variable(bias_name, bias_shape, initializer=initialize_convolutions(w, h, input_channels))

		weight = tf.compat.v1.get_variable(weight_name, weight_shape, initializer=initialize_convolutions(w, h, input_channels))
		bias   = tf.compat.v1.get_variable(bias_name, bias_shape, initializer=initialize_convolutions(w, h, input_channels))

		return weight, bias


	def conv2d(self, x, W, stride):
		layers = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding = "VALID")
		return layers



	def get2d_deconv_output_size(self, input_height, input_width, filter_height, filter_width, stride, padding_type):
		if padding_type == 'VALID':
			out_height = (input_height - 1) * stride + filter_height
			out_width  = (input_width  - 1) * stride + filter_width

		elif padding_type == 'SAME':
			out_height = input_height * stride
			out_width  = input_width  * stride

		return out_height, out_width


	def deconv2d(self, x, W, input_width, input_height, stride):
		filter_height = W.get_shape()[0]	
		filter_width  = W.get_shape()[1]	
		out_channel   = W.get_shape()[2]	

		out_height, out_width = self.get2d_deconv_output_size(input_height, input_width, filter_height, filter_width, stride, 'VALID')
		batch_size = tf.shape(x)[0]
		output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
		return tfnn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

