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
		self.fore_centers = tf.get_variable("fore_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.reduction_dim)))

		self.back_centers = tf.get_variable("back_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.reduction_dim)))


		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		self.fore_W = tf.get_variable("fore_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)))

		self.fore_b = tf.get_variable("fore_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		

		self.back_W = tf.get_variable("back_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)))

		self.back_b = tf.get_variable("back_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))

	def encoder(self):
		
		timesteps = self.enc_in_shape[1]
		# # reduction
		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		# input_feature = tf.nn.relu(input_feature)

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


		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])

		
		mean_feature = tf.reduce_mean(input_feature,axis=-1, keep_dims=True)
		print(mean_feature.get_shape().as_list())

		mean_feature = tf.tile(mean_feature,[1,1,self.centers_num])

		mean_threshold = tf.reduce_mean(mean_feature)
		print(mean_threshold.get_shape().as_list())
		# foreground assignment

		fore_assignment = tf.where(mean_feature>mean_threshold,assignment,tf.zeros_like(assignment))

		# background assignment
		back_assignment = tf.where(mean_feature<=mean_threshold, assignment, tf.zeros_like(assignment))

		# for alpha * c
		fore_a_sum = tf.reduce_sum(fore_assignment,-2,keep_dims=True)
		back_a_sum = tf.reduce_sum(back_assignment,-2,keep_dims=True)

		fore_c = tf.multiply(fore_a_sum,self.fore_centers)
		back_c = tf.multiply(back_a_sum,self.back_centers)

		fore_assignment = tf.transpose(fore_assignment,perm=[0,2,1])
		back_assignment = tf.transpose(back_assignment,perm=[0,2,1])

		def vlad_procedure(assignment, a):
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
			return vlad

		fore_vlad = vlad_procedure(fore_assignment,fore_c)
		back_vlad = vlad_procedure(back_assignment,back_c)

		fore_train_output = tf.nn.xw_plus_b(tf.nn.dropout(fore_vlad,self.dropout),self.fore_W, self.fore_b)

		fore_test_output = tf.nn.xw_plus_b(fore_vlad,self.fore_W, self.fore_b)


		back_train_output = tf.nn.xw_plus_b(tf.nn.dropout(back_vlad,self.dropout),self.back_W, self.back_b)

		back_test_output = tf.nn.xw_plus_b(back_vlad,self.back_W, self.back_b)

		train_output = (fore_train_output+back_train_output)/2.0
		test_output = (fore_test_output+back_test_output)/2.0

		return train_output, test_output
		# return fore_train_output, fore_test_output
		# return back_train_output, back_test_output

	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		train_output, test_output = self.encoder()
		return train_output, test_output

class SeqVladFBModel_v1(object):
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
		# self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))
		self.redu_b = tf.get_variable("redu_b", shape=[self.reduction_dim], initializer=tf.zeros_initializer([self.reduction_dim]))

		self.sep_W = tf.get_variable("sep_W", shape=[self.redu_filter_size, self.redu_filter_size, self.enc_in_shape[-1], 1], 
										initializer=tf.contrib.layers.xavier_initializer())
		# self.sep_b = tf.get_variable("sep_b",initializer=tf.random_normal([1],stddev=1./math.sqrt(1)))
		self.sep_b = tf.get_variable("sep_b", shape=[1], initializer=tf.zeros_initializer([1]))



		self.W_e = tf.get_variable("W_e", shape=[self.filter_size, self.filter_size, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		# self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
		self.b_e = tf.get_variable("b_e", shape=[self.centers_num], initializer=tf.zeros_initializer([self.centers_num]))
		
		self.fore_centers = tf.get_variable("fore_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

		self.back_centers = tf.get_variable("back_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))


		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		self.fore_W = tf.get_variable("fore_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)))

		# self.fore_b = tf.get_variable("fore_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		self.fore_b = tf.get_variable("fore_b", shape=[self.num_class], initializer=tf.zeros_initializer([self.num_class]))
		

		self.back_W = tf.get_variable("back_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)))

		# self.back_b = tf.get_variable("back_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		self.back_b = tf.get_variable("back_b", shape=[self.num_class], initializer=tf.zeros_initializer([self.num_class]))


	def encoder(self):
		
		timesteps = self.enc_in_shape[1]
		# # reduction
		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		weight_feature = tf.add(tf.nn.conv2d(input_feature, self.sep_W, self.stride, self.pad, name='weight_wx'),tf.reshape(self.sep_b,[1, 1, 1, 1]))
		weight_feature = weight_feature*tf.nn.tanh(weight_feature)

		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		# input_feature = tf.nn.relu(input_feature)

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
			
			r = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='r'))
			z = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='z'))

			hh = tf.nn.tanh(assign_t+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))

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


		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])

		
		# mean_feature = tf.reduce_mean(input_feature,axis=-1, keep_dims=True)
		# print(mean_feature.get_shape().as_list())
		weight_feature = tf.reshape(weight_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],1])
		mean_feature = weight_feature
		mean_feature = tf.tile(mean_feature,[1,1,self.centers_num])

		mean_threshold = tf.reduce_mean(mean_feature)
		print(mean_threshold.get_shape().as_list())
		# foreground assignment

		fore_assignment = tf.where(mean_feature>=mean_threshold,assignment,tf.zeros_like(assignment))

		# background assignment
		back_assignment = tf.where(mean_feature<mean_threshold, assignment, tf.zeros_like(assignment))

		# for alpha * c
		fore_a_sum = tf.reduce_sum(fore_assignment,-2,keep_dims=True)
		back_a_sum = tf.reduce_sum(back_assignment,-2,keep_dims=True)

		fore_c = tf.multiply(fore_a_sum,self.fore_centers)
		back_c = tf.multiply(back_a_sum,self.back_centers)

		fore_assignment = tf.transpose(fore_assignment,perm=[0,2,1])
		back_assignment = tf.transpose(back_assignment,perm=[0,2,1])

		def vlad_procedure(assignment, a):
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
			return vlad

		fore_vlad = vlad_procedure(fore_assignment,fore_c)
		back_vlad = vlad_procedure(back_assignment,back_c)

		fore_train_output = tf.nn.xw_plus_b(tf.nn.dropout(fore_vlad,self.dropout),self.fore_W, self.fore_b)

		fore_test_output = tf.nn.xw_plus_b(fore_vlad,self.fore_W, self.fore_b)


		back_train_output = tf.nn.xw_plus_b(tf.nn.dropout(back_vlad,self.dropout),self.back_W, self.back_b)

		back_test_output = tf.nn.xw_plus_b(back_vlad,self.back_W, self.back_b)

		train_output = 0.5*fore_train_output+0.5*back_train_output
		test_output = 0.5*fore_train_output+0.5*back_train_output

		return train_output, test_output
		# return fore_train_output, fore_test_output
		# return back_train_output, back_test_output
		
	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		return self.encoder()


class SeqVladFBModel_v2(object):
	'''
		caption model for ablation studying
		output_dim = num_of_filter
	'''
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
		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)

		self.redu_W = tf.get_variable("redu_W", shape=[self.redu_filter_size, self.redu_filter_size, self.enc_in_shape[-1], self.reduction_dim], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.redu_b = tf.get_variable("redu_b", shape=[self.reduction_dim], initializer=tf.zeros_initializer([self.reduction_dim]))

		self.sep_W = tf.get_variable("sep_W", shape=[self.redu_filter_size, self.redu_filter_size, self.enc_in_shape[-1], 1], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.sep_b = tf.get_variable("sep_b",shape=[1], initializer=tf.zeros_initializer([1]))

		### foreground
		self.fore_W_e = tf.get_variable("fore_W_e", shape=[self.filter_size, self.filter_size, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		# self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
		self.fore_b_e = tf.get_variable("fore_b_e",shape=[self.centers_num], initializer=tf.zeros_initializer([self.centers_num]))

		
		
		self.fore_U_e_r = tf.get_variable("fore_U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.fore_U_e_z = tf.get_variable("fore_U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.fore_U_e_h = tf.get_variable("fore_U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		

		self.fore_centers = tf.get_variable("fore_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

		self.fore_W = tf.get_variable("fore_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)),
				collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.REGULARIZATION_LOSSES])

		self.fore_b = tf.get_variable("fore_b",shape=[self.num_class], initializer=tf.zeros_initializer([self.num_class]))
		
		###  background

		self.back_W_e = tf.get_variable("back_W_e", shape=[self.filter_size, self.filter_size, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		# self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
		self.back_b_e = tf.get_variable("back_b_e",shape=[self.centers_num], initializer=tf.zeros_initializer([self.centers_num]))

		
		
		self.back_U_e_r = tf.get_variable("back_U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.back_U_e_z = tf.get_variable("back_U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.back_U_e_h = tf.get_variable("back_U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		self.back_centers = tf.get_variable("back_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

		self.back_W = tf.get_variable("back_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)),
				collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.REGULARIZATION_LOSSES])

		self.back_b = tf.get_variable("back_b",shape=[self.num_class], initializer=tf.zeros_initializer([self.num_class]))


	def encoder(self):
		
		timesteps = self.enc_in_shape[1]

		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])

		weight_feature = tf.add(tf.nn.conv2d(input_feature, self.sep_W, self.stride, self.pad, name='weight_wx'),
			tf.reshape(self.sep_b,[1, 1, 1, 1]))
		weight_feature = weight_feature*tf.nn.tanh(weight_feature)

		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),
			tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		# input_feature = tf.nn.relu(input_feature)
		# input_feature = tf.nn.l2_normalize(input_feature, -1, name='FeatureNorm')
		self.enc_in_shape = input_feature.get_shape().as_list()

		def get_assignment(W_e, b_e, U_e_r, U_e_z, U_e_h):
			assignment = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
			assignment = tf.add(tf.nn.conv2d(assignment, W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(b_e,[1, 1, 1, self.centers_num]))
			
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
				
				r = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, U_e_r, self.stride, self.pad, name='r'))
				z = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, U_e_z, self.stride, self.pad, name='z'))

				hh = tf.nn.tanh(assign_t+ tf.nn.conv2d(r*h_tm1, U_e_h, self.stride, self.pad, name='uh_hh'))

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

			return assignment
		# assignment = tf.nn.softmax(assignment,dim=-1)

		fore_assignment = get_assignment(self.fore_W_e, self.fore_b_e, self.fore_U_e_r, self.fore_U_e_z, self.fore_U_e_h)
		back_assignment = get_assignment(self.back_W_e, self.back_b_e, self.back_U_e_r, self.back_U_e_z, self.back_U_e_h)

		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])

		
		mean_feature = tf.reduce_mean(input_feature,axis=-1, keep_dims=True)
		print(mean_feature.get_shape().as_list())

		mean_feature = tf.tile(mean_feature,[1,1,self.centers_num])
		mean_threshold = tf.reduce_mean(mean_feature)

		# foreground assignment
		fore_assignment = tf.where(mean_feature>=mean_threshold, fore_assignment, tf.zeros_like(fore_assignment))

		# background assignment
		back_assignment = tf.where(mean_feature<mean_threshold, back_assignment, tf.zeros_like(back_assignment))

		# for alpha * c
		fore_a_sum = tf.reduce_sum(fore_assignment,-2,keep_dims=True)
		back_a_sum = tf.reduce_sum(back_assignment,-2,keep_dims=True)

		fore_c = tf.multiply(fore_a_sum,self.fore_centers)
		back_c = tf.multiply(back_a_sum,self.back_centers)

		fore_assignment = tf.transpose(fore_assignment,perm=[0,2,1])
		back_assignment = tf.transpose(back_assignment,perm=[0,2,1])

		def vlad_procedure(assignment, a):
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
			return vlad

		fore_vlad = vlad_procedure(fore_assignment,fore_c)
		back_vlad = vlad_procedure(back_assignment,back_c)

		fore_train_output = tf.nn.xw_plus_b(tf.nn.dropout(fore_vlad,self.dropout),self.fore_W, self.fore_b)

		fore_test_output = tf.nn.xw_plus_b(fore_vlad,self.fore_W, self.fore_b)


		back_train_output = tf.nn.xw_plus_b(tf.nn.dropout(back_vlad,self.dropout),self.back_W, self.back_b)

		back_test_output = tf.nn.xw_plus_b(back_vlad,self.back_W, self.back_b)

		train_output = (fore_train_output+back_train_output)*0.5
		test_output = (fore_test_output+back_test_output)*0.5
		# train_output = tf.where(fore_train_output>back_train_output,fore_train_output,back_train_output)
		# test_output = tf.where(fore_test_output>back_test_output,fore_test_output,back_test_output)

		return fore_train_output, fore_test_output, back_train_output, back_test_output, train_output, test_output
		# return fore_train_output, fore_test_output
		# return back_train_output, back_test_output



	

	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		fore_train_output, fore_test_output, back_train_output, back_test_output, train_output, test_output = self.encoder()
		return fore_train_output, fore_test_output, back_train_output, back_test_output, train_output, test_output





class SeqVladFBModel_v3(object):
	'''
		caption model for ablation studying
		output_dim = num_of_filter
	'''
	def __init__(self, input_feature,
		num_class=51,
		attention_size=50,
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
		self.attention_size=attention_size

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
		# self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))
		self.redu_b = tf.get_variable("redu_b", shape=[self.reduction_dim], initializer=tf.zeros_initializer([self.reduction_dim]))

		self.attend_U = tf.get_variable("attend_u", shape=[self.enc_in_shape[-1], self.attention_size], 
										initializer=tf.contrib.layers.xavier_initializer())
		

		self.attend_w = tf.get_variable("attend_w", shape=[self.enc_in_shape[-1], self.attention_size], 
										initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.enc_in_shape[-1])))
		self.attend_b = tf.get_variable("attend_b", shape=[self.attention_size], initializer=tf.zeros_initializer([self.attention_size]))

		self.attend_wtanh = tf.get_variable("attend_wtanh", shape=[self.attention_size,1], 
										initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.attention_size)))

		self.W_e = tf.get_variable("W_e", shape=[self.filter_size, self.filter_size, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		# self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
		self.b_e = tf.get_variable("b_e", shape=[self.centers_num], initializer=tf.zeros_initializer([self.centers_num]))
		
		self.fore_centers = tf.get_variable("fore_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

		self.back_centers = tf.get_variable("back_centers",[1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))


		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		self.fore_W = tf.get_variable("fore_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)),
				collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.REGULARIZATION_LOSSES])

		# self.fore_b = tf.get_variable("fore_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		self.fore_b = tf.get_variable("fore_b", shape=[self.num_class], initializer=tf.zeros_initializer([self.num_class]))
		

		self.back_W = tf.get_variable("back_W",[self.centers_num*self.reduction_dim, self.num_class],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.centers_num*self.reduction_dim)),
				collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.REGULARIZATION_LOSSES])

		# self.back_b = tf.get_variable("back_b",initializer=tf.random_normal([self.num_class],stddev=1./math.sqrt(self.num_class)))
		self.back_b = tf.get_variable("back_b", shape=[self.num_class], initializer=tf.zeros_initializer([self.num_class]))


	def encoder(self):
		
		timesteps = self.enc_in_shape[1]
		# # reduction
		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		
		# weight_feature = tf.add(tf.nn.conv2d(input_feature, self.sep_W, self.stride, self.pad, name='weight_wx'),tf.reshape(self.sep_b,[1, 1, 1, 1]))
		
		attend_ux = tf.matmul(tf.reshape(input_feature,[-1,self.enc_in_shape[-1]]),self.attend_U)
		
		global_feat = tf.nn.max_pool(input_feature,[1,self.enc_in_shape[2],self.enc_in_shape[3],1],[1,1,1,1],
			padding='VALID',name='global_pooling')

		print(global_feat.get_shape().as_list())
		global_feat = tf.reshape(global_feat,[-1,self.enc_in_shape[-1]])
		attend_global_feat = tf.nn.xw_plus_b(global_feat,self.attend_w,self.attend_b)

		attend_global_feat = tf.add(tf.reshape(attend_ux,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.attention_size]),
			tf.reshape(attend_global_feat,[-1,self.enc_in_shape[1],1,1,self.attention_size]))
		weight_feature = tf.matmul(tf.nn.tanh(tf.reshape(attend_global_feat,[-1,self.attention_size])),self.attend_wtanh)

		weight_feature = tf.reshape(weight_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],1])


	

		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		# input_feature = tf.nn.relu(input_feature)

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
			
			r = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='r'))
			z = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='z'))

			hh = tf.nn.tanh(assign_t+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))

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


		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])

		
		# mean_feature = tf.reduce_mean(input_feature,axis=-1, keep_dims=True)
		# print(mean_feature.get_shape().as_list())
		weight_feature = tf.reshape(weight_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],1])
		mean_feature = weight_feature
		mean_feature = tf.tile(mean_feature,[1,1,self.centers_num])

		mean_threshold = tf.reduce_mean(mean_feature)
		print(mean_threshold.get_shape().as_list())
		# foreground assignment

		fore_assignment = tf.where(mean_feature>=mean_threshold,assignment,tf.zeros_like(assignment))

		# background assignment
		back_assignment = tf.where(mean_feature<mean_threshold, assignment, tf.zeros_like(assignment))

		# for alpha * c
		fore_a_sum = tf.reduce_sum(fore_assignment,-2,keep_dims=True)
		back_a_sum = tf.reduce_sum(back_assignment,-2,keep_dims=True)

		fore_c = tf.multiply(fore_a_sum,self.fore_centers)
		back_c = tf.multiply(back_a_sum,self.back_centers)

		fore_assignment = tf.transpose(fore_assignment,perm=[0,2,1])
		back_assignment = tf.transpose(back_assignment,perm=[0,2,1])

		def vlad_procedure(assignment, a):
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
			return vlad

		fore_vlad = vlad_procedure(fore_assignment,fore_c)
		back_vlad = vlad_procedure(back_assignment,back_c)

		fore_train_output = tf.nn.xw_plus_b(tf.nn.dropout(fore_vlad,self.dropout),self.fore_W, self.fore_b)

		fore_test_output = tf.nn.xw_plus_b(fore_vlad,self.fore_W, self.fore_b)


		back_train_output = tf.nn.xw_plus_b(tf.nn.dropout(back_vlad,self.dropout),self.back_W, self.back_b)

		back_test_output = tf.nn.xw_plus_b(back_vlad,self.back_W, self.back_b)

		train_output = 0.5*fore_train_output+0.5*back_train_output
		test_output = 0.5*fore_train_output+0.5*back_train_output

		return train_output, test_output
		# return fore_train_output, fore_test_output
		# return back_train_output, back_test_output
		
	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		return self.encoder()