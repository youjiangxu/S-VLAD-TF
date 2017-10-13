import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import tensorflow as tf

import numpy as np
import math

rng = np.random
rng.seed(1234)

def hard_sigmoid(x):
	x = (0.2 * x) + 0.5
	x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),tf.cast(1., dtype=tf.float32))
	return x

class NetVladModel(object):
	'''
		caption model for ablation studying
		output_dim = num_of_filter
	'''
	def __init__(self, input_feature,
		num_class=51,
		reduction_dim=512,
		centers_num=16,
		dropout=0.5,
		redu_filter_size=3,
		filter_size=1, stride=[1,1,1,1], pad='SAME', 
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):

		self.reduction_dim=reduction_dim
		

		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

		self.num_class=num_class
		self.filter_size = filter_size
		self.stride = stride
		self.pad = pad

		self.dropout=dropout
		self.centers_num = centers_num

		self.redu_filter_size = redu_filter_size

		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences

		self.enc_in_shape = self.input_feature.get_shape().as_list()


	def init_parameters(self):
		print('init_parameters ...')


		self.redu_W = tf.get_variable("redu_W", shape=[self.redu_filter_size, self.redu_filter_size, self.enc_in_shape[-1], self.reduction_dim], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))

		

		self.W_e = tf.get_variable("W_e", shape=[self.filter_size, self.filter_size, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
		self.centers = tf.get_variable("centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))



		# classification parameters
		self.liner_W = tf.get_variable("liner_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)))

		self.liner_b = tf.get_variable("liner_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		


	def encoder(self):
		
		timesteps = self.enc_in_shape[1]

		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		input_feature = tf.nn.relu(input_feature)

		self.enc_in_shape = input_feature.get_shape().as_list()

		
		assignment = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, self.centers_num]))
		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])
		assignment = tf.nn.softmax(assignment,dim=-1)

		# for alpha * c
		a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
		a = tf.multiply(a_sum,self.centers)

		# for alpha * x
		assignment = tf.transpose(assignment,perm=[0,2,1])

		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.enc_in_shape[4]])

		vlad = tf.matmul(assignment,input_feature)
		vlad = tf.transpose(vlad, perm=[0,2,1])

		# for differnce
		vlad = tf.subtract(vlad,a)
		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1],self.centers_num])

		vlad = tf.reduce_sum(vlad,axis=1)

		vlad = tf.nn.l2_normalize(vlad,1)

		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1]*self.centers_num])

		vlad = tf.nn.l2_normalize(vlad,-1)


		train_output = tf.nn.xw_plus_b(tf.nn.dropout(vlad,self.dropout),self.liner_W, self.liner_b)
		test_output = tf.nn.xw_plus_b(vlad,self.liner_W, self.liner_b)

		return train_output, test_output
	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		train_output, test_output = self.encoder()
		return train_output, test_output





class SeqVladWithReduModel(object):
	'''
		caption model for ablation studying
		output_dim = num_of_filter
	'''
	def __init__(self, input_feature,
		num_class=51,
		reduction_dim=512,
		centers_num=16,
		dropout=0.5,
		redu_filter_size=3,
		filter_size=1, stride=[1,1,1,1], pad='SAME', 
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):

		self.reduction_dim=reduction_dim
		

		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

		self.num_class=num_class
		self.filter_size = filter_size
		self.stride = stride
		self.pad = pad

		self.dropout=dropout
		self.redu_filter_size=redu_filter_size
		self.centers_num = centers_num


		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences

		self.enc_in_shape = self.input_feature.get_shape().as_list()

	def init_parameters(self):
		print('init_parameters ...')

		# encoder parameters
		# print(self.enc_in_shape)

		self.redu_W = tf.get_variable("redu_W", shape=[self.redu_filter_size, self.redu_filter_size, self.enc_in_shape[-1], self.reduction_dim], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))

		
		self.W_e = tf.get_variable("W_e", shape=[self.filter_size, self.filter_size, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
		self.centers = tf.get_variable("centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))



		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		self.liner_W = tf.get_variable("liner_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)))

		self.liner_b = tf.get_variable("liner_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		

	def encoder(self):
		
		timesteps = self.enc_in_shape[1]
		# # reduction
		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		input_feature = tf.nn.relu(input_feature)

		self.enc_in_shape = input_feature.get_shape().as_list()

		assignment = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, self.centers_num]))
		
		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.centers_num])



		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
		assignment = tf.transpose(assignment, perm=axis)

		input_assignment = tf.TensorArray(
	            dtype=assignment.dtype,
	            size=timesteps,
	            tensor_array_name='input_assignment')
		if hasattr(input_assignment, 'unstack'):
			input_assignment = input_assignment.unstack(assignment)
		else:
			input_assignment = input_assignment.unpack(assignment)	

		hidden_states = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states')

		def get_init_state(x, output_dims):
			initial_state = tf.zeros_like(x)
			initial_state = tf.reduce_sum(initial_state,axis=[1,4])
			initial_state = tf.expand_dims(initial_state,dim=-1)
			initial_state = tf.tile(initial_state,[1,1,1,output_dims])
			return initial_state
		def step(time, hidden_states, h_tm1):
			assign_t = input_assignment.read(time) # batch_size * dim
			
			r = hard_sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='r'))
			z = hard_sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='z'))

			hh = tf.tanh(assign_t+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))

			h = (1-z)*hh + z*h_tm1
			
			hidden_states = hidden_states.write(time, h)

			return (time+1,hidden_states, h)

		time = tf.constant(0, dtype='int32', name='time')
		initial_state = get_init_state(input_feature,self.centers_num)

		feature_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=step,
	            loop_vars=(time, hidden_states, initial_state ),
	            parallel_iterations=32,
	            swap_memory=True)


		hidden_states = feature_out[-2]
		if hasattr(hidden_states, 'stack'):
			assignment = hidden_states.stack()
		else:
			assignment = hidden_states.pack()

		
		
		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
		assignment = tf.transpose(assignment, perm=axis)


		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])

		# assignment = tf.nn.softmax(assignment,dim=-1)

		# for alpha * c
		a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
		a = tf.multiply(a_sum,self.centers)
		# for alpha * x
		assignment = tf.transpose(assignment,perm=[0,2,1])

		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])

		vlad = tf.matmul(assignment,input_feature)
		vlad = tf.transpose(vlad, perm=[0,2,1])
		# for differnce
		vlad = tf.subtract(vlad,a)

		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1], self.enc_in_shape[-1], self.centers_num])
		vlad = tf.reduce_sum(vlad,axis=1)

		vlad = tf.nn.l2_normalize(vlad,1)

		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1]*self.centers_num])
		# vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1]*self.centers_num])

		vlad = tf.nn.l2_normalize(vlad,-1)


		train_output = tf.nn.xw_plus_b(tf.nn.dropout(vlad,self.dropout),self.liner_W, self.liner_b)

		test_output = tf.nn.xw_plus_b(vlad,self.liner_W, self.liner_b)

		return train_output, test_output
	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		train_output, test_output = self.encoder()
		return train_output, test_output


class SeqVladWithReduNotShareModel(object):
	'''
		caption model for ablation studying
		output_dim = num_of_filter
	'''
	def __init__(self, input_feature,
		num_class=51,
		reduction_dim=512,
		centers_num=16,
		dropout=0.5,
		redu_filter_size=3,
		filter_size=1, stride=[1,1,1,1], pad='SAME', 
		inner_activation='hard_sigmoid',activation='tanh',
		return_sequences=True):

		self.reduction_dim=reduction_dim
		

		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

		self.num_class=num_class
		self.filter_size = filter_size
		self.stride = stride
		self.pad = pad

		self.dropout=dropout
		self.centers_num = centers_num

		self.redu_filter_size=redu_filter_size

		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences

		self.enc_in_shape = self.input_feature.get_shape().as_list()
	def init_parameters(self):
		print('init_parameters ...')

		# encoder parameters
		# print(self.enc_in_shape)

		self.redu_W = tf.get_variable("redu_W", shape=[self.redu_filter_size, self.redu_filter_size, self.enc_in_shape[-1], self.reduction_dim], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))

		
		self.W_e = tf.get_variable("W_e", shape=[self.filter_size, self.filter_size, self.reduction_dim, 3*self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([3*self.centers_num],stddev=1./math.sqrt(3*self.centers_num)))
		self.centers = tf.get_variable("centers",[1, 1, 1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))




		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		# classification parameters
		self.liner_W = tf.get_variable("liner_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)))

		self.liner_b = tf.get_variable("liner_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		

		
	
	def encoder(self):
		
		timesteps = self.enc_in_shape[1]
		# # reduction
		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		input_feature = tf.nn.relu(input_feature)

		self.enc_in_shape = input_feature.get_shape().as_list()

		assignment = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, 3*self.centers_num]))
		
		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],3*self.centers_num])



		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
		assignment = tf.transpose(assignment, perm=axis)

		input_assignment = tf.TensorArray(
	            dtype=assignment.dtype,
	            size=timesteps,
	            tensor_array_name='input_assignment')
		if hasattr(input_assignment, 'unstack'):
			input_assignment = input_assignment.unstack(assignment)
		else:
			input_assignment = input_assignment.unpack(assignment)	

		hidden_states = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states')

		def get_init_state(x, output_dims):
			initial_state = tf.zeros_like(x)
			initial_state = tf.reduce_sum(initial_state,axis=[1,4])
			initial_state = tf.expand_dims(initial_state,dim=-1)
			initial_state = tf.tile(initial_state,[1,1,1,output_dims])
			return initial_state
		def step(time, hidden_states, h_tm1):
			assign_t = input_assignment.read(time) # batch_size * dim
			assign_t_r = assign_t[:,:,:,0:self.centers_num]
			assign_t_z = assign_t[:,:,:,self.centers_num:2*self.centers_num]
			assign_t_h = assign_t[:,:,:,2*self.centers_num::]
			
			r = hard_sigmoid(assign_t_r+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='r'))
			z = hard_sigmoid(assign_t_z+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='z'))

			hh = tf.tanh(assign_t_h+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='hh'))

			h = (1-z)*hh + z*h_tm1
			
			hidden_states = hidden_states.write(time, h)

			return (time+1,hidden_states, h)

		time = tf.constant(0, dtype='int32', name='time')
		initial_state = get_init_state(input_feature,self.centers_num)

		feature_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=step,
	            loop_vars=(time, hidden_states, initial_state ),
	            parallel_iterations=32,
	            swap_memory=True)


		hidden_states = feature_out[-2]
		if hasattr(hidden_states, 'stack'):
			assignment = hidden_states.stack()
		else:
			assignment = hidden_states.pack()

		
		
		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
		assignment = tf.transpose(assignment, perm=axis)


		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])

		# for alpha * c
		a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
		a = tf.multiply(a_sum,self.centers)
		# for alpha * x
		assignment = tf.transpose(assignment,perm=[0,2,1])

		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])

		vlad = tf.matmul(assignment,input_feature)
		vlad = tf.transpose(vlad, perm=[0,2,1])
		# for differnce
		vlad = tf.subtract(vlad,a)

		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1], self.enc_in_shape[-1], self.centers_num])
		vlad = tf.reduce_sum(vlad,axis=1)

		vlad = tf.nn.l2_normalize(vlad,1)

		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1]*self.centers_num])
		# vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1]*self.centers_num])

		vlad = tf.nn.l2_normalize(vlad,-1)


		train_output = tf.nn.xw_plus_b(tf.nn.dropout(vlad,self.dropout),self.liner_W, self.liner_b)

		test_output = tf.nn.xw_plus_b(vlad,self.liner_W, self.liner_b)

		return train_output, test_output
	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		train_output, test_output = self.encoder()
		return train_output, test_output
