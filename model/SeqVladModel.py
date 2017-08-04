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

# class AttentionModel(object):
# 	'''
# 		caption model for ablation studying
# 		output_dim = num_of_filter
# 	'''
# 	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, 
# 		filter_height=1, filter_width=1, stride=[1,1,1,1], pad='SAME', 
# 		done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5,
# 		attention_dim = 100, dropout=0.5,
# 		inner_activation='hard_sigmoid',activation='tanh',
# 		return_sequences=True):

# 		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

# 		self.input_captions = input_captions

# 		self.voc_size = voc_size
# 		self.d_w2v = d_w2v

# 		self.output_dim = output_dim
# 		self.filter_height = filter_height
# 		self.filter_width = filter_width
# 		self.stride = stride
# 		self.pad = pad

# 		self.beam_size = beam_size

# 		assert(beamsearch_batchsize==1)
# 		self.batch_size = beamsearch_batchsize
# 		self.done_token = done_token
# 		self.max_len = max_len

# 		self.dropout = dropout

# 		self.inner_activation = inner_activation
# 		self.activation = activation
# 		self.return_sequences = return_sequences
# 		self.attention_dim = attention_dim

# 		self.encoder_input_shape = self.input_feature.get_shape().as_list()
# 		self.decoder_input_shape = self.input_captions.get_shape().as_list()
# 		print('encoder_input_shape', self.encoder_input_shape)
# 	def init_parameters(self):
# 		print('init_parameters ...')

# 		# encoder parameters
# 		# print(self.encoder_input_shape)
# 		encoder_i2h_shape = (self.filter_height, self.filter_width, self.encoder_input_shape[-1], 3*self.output_dim)
# 		encoder_h2h_shape = (self.filter_height, self.filter_width, self.output_dim, self.output_dim)
# 		self.W_e = InitUtil.init_weight_variable(encoder_i2h_shape,init_method='glorot_uniform',name="W_e")
# 		self.b_e = InitUtil.init_bias_variable((3*self.output_dim,),name="b_e")

# 		self.U_e_r = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='glorot_uniform',name="U_e_r")
# 		self.U_e_z = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='glorot_uniform',name="U_e_z")
# 		self.U_e_h = InitUtil.init_weight_variable(encoder_h2h_shape,init_method='glorot_uniform',name="U_e_h")

		



# 		# decoder parameters
# 		self.T_w2v, self.T_mask = self.init_embedding_matrix()

# 		decoder_i2h_shape = (self.d_w2v,self.output_dim)
# 		decoder_h2h_shape = (self.output_dim,self.output_dim)
# 		self.W_d_r = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_r")
# 		self.W_d_z = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_z")
# 		self.W_d_h = InitUtil.init_weight_variable(decoder_i2h_shape,init_method='glorot_uniform',name="W_d_h")

# 		self.U_d_r = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='glorot_uniform',name="U_d_r")
# 		self.U_d_z = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='glorot_uniform',name="U_d_z")
# 		self.U_d_h = InitUtil.init_weight_variable(decoder_h2h_shape,init_method='glorot_uniform',name="U_d_h")

# 		self.b_d_r = InitUtil.init_bias_variable((self.output_dim,),name="b_d_r")
# 		self.b_d_z = InitUtil.init_bias_variable((self.output_dim,),name="b_d_z")
# 		self.b_d_h = InitUtil.init_bias_variable((self.output_dim,),name="b_d_h")


		
# 		self.W_a = InitUtil.init_weight_variable((self.output_dim,self.attention_dim),init_method='glorot_uniform',name="W_a")
# 		self.U_a = InitUtil.init_weight_variable((self.output_dim,self.attention_dim),init_method='glorot_uniform',name="U_a")
# 		self.b_a = InitUtil.init_bias_variable((self.attention_dim,),name="b_a")

# 		self.W = InitUtil.init_weight_variable((self.attention_dim,1),init_method='glorot_uniform',name="W")

# 		self.A_z = InitUtil.init_weight_variable((self.output_dim,self.output_dim),init_method='glorot_uniform',name="A_z")

# 		self.A_r = InitUtil.init_weight_variable((self.output_dim,self.output_dim),init_method='glorot_uniform',name="A_r")

# 		self.A_h = InitUtil.init_weight_variable((self.output_dim,self.output_dim),init_method='glorot_uniform',name="A_h")


# 		# classification parameters
# 		self.W_c = InitUtil.init_weight_variable((self.output_dim,self.voc_size),init_method='glorot_uniform',name='W_c')
# 		self.b_c = InitUtil.init_bias_variable((self.voc_size,),name="b_c")

# 	def init_embedding_matrix(self):
# 		'''init word embedding matrix
# 		'''
# 		voc_size = self.voc_size
# 		d_w2v = self.d_w2v	
# 		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
# 		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

# 		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
# 		for v in range(voc_size):
# 			LUT[v] = rng.randn(d_w2v)
# 			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

# 		# word 0 is blanked out, word 1 is 'UNK'
# 		LUT[0] = np.zeros((d_w2v))
# 		# setup LUT!
# 		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)

# 		return T_w2v, T_mask 

# 	def encoder(self):
# 		'''
# 			visual feature part
# 		'''
# 		print('building encoder ... ...')
# 		def get_init_state(x, output_dims):
# 			initial_state = tf.zeros_like(x)
# 			initial_state = tf.reduce_sum(initial_state,axis=[1,4])
# 			initial_state = tf.expand_dims(initial_state,dim=-1)
# 			initial_state = tf.tile(initial_state,[1,1,1,self.output_dim])
# 			return initial_state

		
		
# 		timesteps = self.encoder_input_shape[1]

# 		embedded_feature = self.input_feature

# 		embedded_feature = tf.reshape(embedded_feature,[-1,self.encoder_input_shape[2],self.encoder_input_shape[3],self.encoder_input_shape[4]])
# 		embedded_feature = tf.add(tf.nn.conv2d(embedded_feature, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[-1, 1, 1, 3*self.output_dim]))
# 		embedded_feature = tf.reshape(embedded_feature,[-1,self.encoder_input_shape[1],self.encoder_input_shape[2],self.encoder_input_shape[3],3*self.output_dim])


# 		initial_state = get_init_state(embedded_feature, self.output_dim)


# 		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
# 		embedded_feature = tf.transpose(embedded_feature, perm=axis)

# 		input_feature = tf.TensorArray(
# 	            dtype=embedded_feature.dtype,
# 	            size=timesteps,
# 	            tensor_array_name='input_feature')
# 		if hasattr(input_feature, 'unstack'):
# 			input_feature = input_feature.unstack(embedded_feature)
# 		else:
# 			input_feature = input_feature.unpack(embedded_feature)	


# 		hidden_states = tf.TensorArray(
# 	            dtype=tf.float32,
# 	            size=timesteps,
# 	            tensor_array_name='hidden_states')

		
# 		def feature_step(time, hidden_states, h_tm1):
# 			preprocess_x = input_feature.read(time) # batch_size * dim
# 			# # (batch, height, width, channels)
			
# 			preprocess_x_r = preprocess_x[:,:,:,0:self.output_dim]
# 			preprocess_x_z = preprocess_x[:,:,:,self.output_dim:2*self.output_dim]
# 			preprocess_x_h = preprocess_x[:,:,:,2*self.output_dim::]

# 			r = hard_sigmoid(preprocess_x_r+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='uh_r'))
# 			z = hard_sigmoid(preprocess_x_z+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='uh_z'))
# 			hh = tf.nn.tanh(preprocess_x_h+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))

			
# 			h = (1-z)*hh + z*h_tm1

			
# 			hidden_states = hidden_states.write(time, h)

# 			return (time+1,hidden_states,h)

		

# 		time = tf.constant(0, dtype='int32', name='time')


# 		feature_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=feature_step,
# 	            loop_vars=(time, hidden_states, initial_state),
# 	            parallel_iterations=32,
# 	            swap_memory=True)

# 		last_output = feature_out[-1] 
# 		hidden_states = feature_out[-2]
# 		if hasattr(hidden_states, 'stack'):
# 			encoder_output = hidden_states.stack()
# 		else:
# 			encoder_output = hidden_states.pack()

		
# 		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
# 		encoder_output = tf.transpose(encoder_output, perm=axis)

# 		encoder_output = tf.reshape(encoder_output,[-1, self.encoder_input_shape[1], self.encoder_input_shape[2], self.encoder_input_shape[3], self.output_dim])
# 		encoder_output = tf.reduce_mean(encoder_output,axis=[2,3])
# 		last_output = tf.reduce_mean(last_output,axis=[1,2])

# 		return last_output, encoder_output

# 	def decoder(self, initial_state, input_feature, conv_memory):
# 		'''
# 			captions: (batch_size x timesteps) ,int32
# 			d_w2v: dimension of word 2 vector
# 		'''
# 		captions = self.input_captions

# 		print('building decoder ... ...')
# 		mask =  tf.not_equal(captions,0)


# 		loss_mask = tf.cast(mask,tf.float32)

# 		embedded_captions = tf.gather(self.T_w2v,captions)*tf.gather(self.T_mask,captions)

# 		timesteps = self.decoder_input_shape[1]


# 		# batch_size x timesteps x dim -> timesteps x batch_size x dim
# 		axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
# 		embedded_captions = tf.transpose(embedded_captions, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



# 		input_embedded_words = tf.TensorArray(
# 	            dtype=embedded_captions.dtype,
# 	            size=timesteps,
# 	            tensor_array_name='input_embedded_words')


# 		if hasattr(input_embedded_words, 'unstack'):
# 			input_embedded_words = input_embedded_words.unstack(embedded_captions)
# 		else:
# 			input_embedded_words = input_embedded_words.unpack(embedded_captions)	


# 		# preprocess mask
# 		mask = tf.expand_dims(mask,dim=-1)
		
# 		mask = tf.transpose(mask,perm=axis)

# 		input_mask = tf.TensorArray(
# 			dtype=mask.dtype,
# 			size=timesteps,
# 			tensor_array_name='input_mask'
# 			)

# 		if hasattr(input_mask, 'unstack'):
# 			input_mask = input_mask.unstack(mask)
# 		else:
# 			input_mask = input_mask.unpack(mask)


# 		train_hidden_state = tf.TensorArray(
# 	            dtype=tf.float32,
# 	            size=timesteps,
# 	            tensor_array_name='train_hidden_state')

# 		def step(x_t,h_tm1):

# 			# ori_feature = tf.reshape(self.input_feature,(-1,self.encoder_input_shape[-1]))

# 			# attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[-2],self.attention_dim))
# 			# attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1),[1,self.encoder_input_shape[-2],1])

# 			# attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
# 			# attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
# 			# # attend_e = tf.reshape(attend_e,(-1,attention_dim))
# 			# attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[-2],1)),dim=1)

# 			# attend_fea = self.input_feature * tf.tile(attend_e,[1,1,self.encoder_input_shape[-1]])
# 			# attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)

# 			ori_feature = tf.reshape(input_feature,(-1,self.output_dim))

# 			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[1],self.attention_dim))
# 			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1),[1,self.encoder_input_shape[1],1])

# 			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
# 			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W) # batch_size * timestep
# 			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
# 			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[1],1)),dim=1)

# 			attend_fea = input_feature * tf.tile(attend_e,[1,1,self.output_dim])
# 			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)


# 			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_d_r, self.b_d_r)
# 			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_d_z, self.b_d_z)
# 			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_d_h, self.b_d_h)

# 			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + tf.matmul(attend_fea,self.A_r))
# 			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + tf.matmul(attend_fea,self.A_z))
# 			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + tf.matmul(attend_fea,self.A_h))

			
# 			h = (1-z)*hh + z*h_tm1


# 			return h

# 		def train_step(time, train_hidden_state, h_tm1):
# 			x_t = input_embedded_words.read(time) # batch_size * dim
# 			mask_t = input_mask.read(time)

# 			h = step(x_t,h_tm1)

# 			tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

# 			h = tf.where(tiled_mask_t, h, h_tm1) # (batch_size, output_dims)
			
# 			train_hidden_state = train_hidden_state.write(time, h)

# 			return (time+1,train_hidden_state,h)

		

# 		time = tf.constant(0, dtype='int32', name='time')


# 		train_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=train_step,
# 	            loop_vars=(time, train_hidden_state, initial_state),
# 	            parallel_iterations=32,
# 	            swap_memory=True)


# 		train_hidden_state = train_out[1]
# 		train_last_output = train_out[-1] 
		
# 		if hasattr(train_hidden_state, 'stack'):
# 			train_outputs = train_hidden_state.stack()
# 		else:
# 			train_outputs = train_hidden_state.pack()

# 		axis = [1,0] + list(range(2,3))
# 		train_outputs = tf.transpose(train_outputs,perm=axis)



# 		train_outputs = tf.reshape(train_outputs,(-1,self.output_dim))
# 		train_outputs = tf.nn.dropout(train_outputs, self.dropout)
# 		predict_score = tf.matmul(train_outputs,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
# 		predict_score = tf.reshape(predict_score,(-1,timesteps,self.voc_size))
# 		# predict_score = tf.nn.softmax(predict_score,-1)
# 		# test phase


# 		test_input_embedded_words = tf.TensorArray(
# 	            dtype=embedded_captions.dtype,
# 	            size=timesteps+1,
# 	            tensor_array_name='test_input_embedded_words')

# 		predict_words = tf.TensorArray(
# 	            dtype=tf.int64,
# 	            size=timesteps,
# 	            tensor_array_name='predict_words')

# 		test_hidden_state = tf.TensorArray(
# 	            dtype=tf.float32,
# 	            size=timesteps,
# 	            tensor_array_name='test_hidden_state')
# 		test_input_embedded_words = test_input_embedded_words.write(0,embedded_captions[0])

# 		def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1):
# 			x_t = test_input_embedded_words.read(time) # batch_size * dim

# 			h = step(x_t,h_tm1)

# 			test_hidden_state = test_hidden_state.write(time, h)


# 			# drop_h = tf.nn.dropout(h, 0.5)
# 			drop_h = h*self.dropout
# 			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

# 			# predict_score_t = tf.matmul(normed_h,tf.transpose(T_w2v,perm=[1,0]))
# 			predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
# 			predict_word_t = tf.argmax(predict_score_t,-1)

# 			predict_words = predict_words.write(time, predict_word_t) # output


# 			predict_word_t = tf.gather(self.T_w2v,predict_word_t)*tf.gather(self.T_mask,predict_word_t)

# 			test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

# 			return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h)


# 		time = tf.constant(0, dtype='int32', name='time')


# 		test_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=test_step,
# 	            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state),
# 	            parallel_iterations=32,
# 	            swap_memory=True)


# 		predict_words = test_out[-2]
		
# 		if hasattr(predict_words, 'stack'):
# 			predict_words = predict_words.stack()
# 		else:
# 			predict_words = predict_words.pack()

# 		axis = [1,0] + list(range(2,3))

# 		predict_words = tf.transpose(predict_words,perm=[1,0])
# 		predict_words = tf.reshape(predict_words,(-1,timesteps))

# 		return predict_score, predict_words, loss_mask


# 	def beamSearchDecoder(self, initial_state_h, input_feature):
# 		'''
# 			captions: (batch_size x timesteps) ,int32
# 			d_w2v: dimension of word 2 vector
# 		'''

# 		# self.batch_size = self.input_captions.get_shape().as_list().eval()[0]
# 		def step(x_t,h_tm1):
# 			ori_feature = tf.tile(tf.expand_dims(input_feature,dim=1),[1,self.beam_size,1,1])
# 			ori_feature = tf.reshape(ori_feature,(-1,self.output_dim))

# 			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.encoder_input_shape[1],self.attention_dim))
# 			attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1),[1,self.encoder_input_shape[1],1])

# 			attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
# 			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)# batch_size * timestep
# 			# attend_e = tf.reshape(attend_e,(-1,attention_dim))
# 			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.encoder_input_shape[1],1)),dim=1)
# 			print('attend_e.get_shape()',attend_e.get_shape().as_list())

# 			attend_fea = tf.multiply(tf.reshape(ori_feature,[self.batch_size,self.beam_size,self.encoder_input_shape[1],self.output_dim]),
# 				tf.reshape(attend_e,[self.batch_size,self.beam_size,self.encoder_input_shape[1],1]))
# 			attend_fea = tf.reshape(tf.reduce_sum(attend_fea,reduction_indices=2),[self.batch_size*self.beam_size,self.output_dim])


			
# 			preprocess_x_r = tf.nn.xw_plus_b(x_t, self.W_d_r, self.b_d_r)
# 			preprocess_x_z = tf.nn.xw_plus_b(x_t, self.W_d_z, self.b_d_z)
# 			preprocess_x_h = tf.nn.xw_plus_b(x_t, self.W_d_h, self.b_d_h)

# 			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + tf.matmul(attend_fea,self.A_r))
# 			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + tf.matmul(attend_fea,self.A_z))
# 			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + tf.matmul(attend_fea,self.A_h))

			
# 			h = (1-z)*hh + z*h_tm1
			
# 			return h
# 		def take_step_zero(x_0, h_0):

# 			x_0 = tf.gather(self.T_w2v,x_0)*tf.gather(self.T_mask,x_0)
# 			x_0 = tf.reshape(x_0,[self.batch_size*self.beam_size,self.d_w2v])
# 			h = step(x_0,h_0)
# 			drop_h = h
# 			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
# 			# logprobs = tf.log(tf.nn.softmax(predict_score_t))
# 			logprobs = tf.nn.log_softmax(predict_score_t)

# 			print('logrobs.get_shape().as_list():',logprobs.get_shape().as_list())

# 			logprobs_batched = tf.reshape(logprobs, [-1, self.beam_size, self.voc_size])

			
# 			past_logprobs, indices = tf.nn.top_k(
# 			        logprobs_batched[:,0,:],self.beam_size)

# 			symbols = indices % self.voc_size
# 			parent_refs = indices//self.voc_size
# 			h = tf.gather(h,  tf.reshape(parent_refs,[-1]))
# 			print('symbols.shape',symbols.get_shape().as_list())

# 			past_symbols = tf.concat([tf.expand_dims(symbols, 2), tf.zeros((self.batch_size, self.beam_size, self.max_len-1), dtype=tf.int32)],-1)
# 			return symbols, h, past_symbols, past_logprobs


# 		def test_step(time, x_t, h_tm1, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):

# 			x_t = tf.gather(self.T_w2v,x_t)*tf.gather(self.T_mask,x_t)
# 			x_t = tf.reshape(x_t,[self.batch_size*self.beam_size,self.d_w2v])
# 			h = step(x_t,h_tm1)

# 			print('h.shape()',h.get_shape().as_list())
# 			drop_h = h
# 			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

# 			logprobs = tf.nn.log_softmax(predict_score_t)
# 			logprobs = tf.reshape(logprobs, [1, self.beam_size, self.voc_size])

		
# 			logprobs = logprobs+tf.expand_dims(past_logprobs, 2)
# 			past_logprobs, topk_indices = tf.nn.top_k(
# 			    tf.reshape(logprobs, [1, self.beam_size * self.voc_size]),
# 			    self.beam_size, 
# 			    sorted=False
# 			)       

# 			symbols = topk_indices % self.voc_size
# 			symbols = tf.reshape(symbols, [1,self.beam_size])
# 			parent_refs = topk_indices // self.voc_size


# 			h = tf.gather(h,  tf.reshape(parent_refs,[-1]))
# 			past_symbols_batch_major = tf.reshape(past_symbols[:,:,0:time], [-1, time])

# 			beam_past_symbols = tf.gather(past_symbols_batch_major,  parent_refs)
			

# 			past_symbols = tf.concat([beam_past_symbols, tf.expand_dims(symbols, 2), tf.zeros((1, self.beam_size, self.max_len-time-1), dtype=tf.int32)],2)
# 			past_symbols = tf.reshape(past_symbols, [1,self.beam_size,self.max_len])
			
# 			# For finishing the beam here
# 			cond1 = tf.equal(symbols,tf.ones_like(symbols,tf.int32)*self.done_token) # condition on done sentence
			

# 			for_finished_logprobs = tf.where(cond1,past_logprobs,tf.ones_like(past_logprobs,tf.float32)* -1e5)

# 			done_indice_max = tf.cast(tf.argmax(for_finished_logprobs,axis=-1),tf.int32)
# 			logprobs_done_max = tf.reduce_max(for_finished_logprobs,reduction_indices=-1)

			
# 			done_past_symbols = tf.gather(tf.reshape(past_symbols,[self.beam_size,self.max_len]),done_indice_max)
# 			# # improved beamsearch method 
# 			logprobs_done_max = tf.div(-logprobs_done_max,tf.cast(time,tf.float32))
# 			cond2 = tf.greater(logprobs_finished_beams,logprobs_done_max)

# 			cond3 = tf.equal(done_past_symbols[:,time],self.done_token)
# 			cond4 = tf.equal(time,self.max_len-1)

# 			finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
# 			                                done_past_symbols,
# 			                                finished_beams)
# 			logprobs_finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
# 											logprobs_done_max, 
# 											logprobs_finished_beams)

			

# 			return (time+1, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)



# 		captions = self.input_captions

# 		# past_logprobs = tf.ones((self.batch_size,), dtype=tf.float32) * -1e5
# 		# past_symbols = tf.zeros((self.batch_size, self.beam_size, self.max_len), dtype=tf.int32)

# 		finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32)
# 		logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * float('inf')

# 		x_0 = captions[:,0]
# 		x_0 = tf.expand_dims(x_0,dim=-1)
# 		print('x_0',x_0.get_shape().as_list())
# 		x_0 = tf.tile(x_0,[1,self.beam_size])


# 		h_0 = tf.expand_dims(initial_state_h,dim=1)
# 		h_0 = tf.reshape(tf.tile(h_0,[1,self.beam_size,1]),[self.batch_size*self.beam_size,self.output_dim])

		

# 		symbols, h, past_symbols, past_logprobs = take_step_zero(x_0, h_0)
# 		time = tf.constant(1, dtype='int32', name='time')
# 		timesteps = self.max_len

		

# 		test_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=test_step,
# 	            loop_vars=(time, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams),
# 	            parallel_iterations=32,
# 	            swap_memory=True)

		

		


# 		out_finished_beams = test_out[-2]
# 		out_logprobs_finished_beams = test_out[-1]
# 		out_past_symbols = test_out[-4]

# 		return   out_finished_beams, out_logprobs_finished_beams, out_past_symbols
# 	def build_model(self):
# 		print('building model ... ...')
# 		self.init_parameters()
# 		last_output, encoder_output = self.encoder()
# 		predict_score, predict_words , loss_mask= self.decoder(last_output, encoder_output)
# 		finished_beam, logprobs_finished_beams, past_symbols = self.beamSearchDecoder(last_output, encoder_output)
# 		return predict_score, predict_words, loss_mask, finished_beam, logprobs_finished_beams, past_symbols







# class NetVladAttentionModel(object):
# 	'''
# 		caption model for ablation studying
# 		output_dim = num_of_filter
# 	'''
# 	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim,
# 		reduction_dim=512, 
# 		centers_num=16,
# 		filter_size=1, stride=[1,1,1,1], pad='SAME', 
# 		done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5,
# 		attention_dim = 100, dropout=0.5,
# 		inner_activation='hard_sigmoid',activation='tanh',
# 		return_sequences=True):

# 		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

# 		self.input_captions = input_captions

# 		self.voc_size = voc_size
# 		self.d_w2v = d_w2v

# 		self.output_dim = output_dim
# 		self.filter_size = filter_size
# 		self.stride = stride
# 		self.pad = pad

# 		self.centers_num = centers_num

# 		self.reduction_dim = reduction_dim
# 		# self.init_w = init_w
# 		# self.init_b = init_b
# 		# self.init_centers = init_centers

# 		self.beam_size = beam_size

# 		assert(beamsearch_batchsize==1)
# 		self.batch_size = beamsearch_batchsize
# 		self.done_token = done_token
# 		self.max_len = max_len

# 		self.dropout = dropout



# 		self.inner_activation = inner_activation
# 		self.activation = activation
# 		self.return_sequences = return_sequences
# 		self.attention_dim = attention_dim

# 		self.enc_in_shape = self.input_feature.get_shape().as_list()
# 		self.decoder_input_shape = self.input_captions.get_shape().as_list()
# 		print('enc_in_shape', self.enc_in_shape)
# 	def init_parameters(self):
# 		print('init_parameters ...')


# 		self.redu_W = tf.get_variable("redu_W", shape=[3, 3, self.enc_in_shape[-1], self.reduction_dim], 
# 										initializer=tf.contrib.layers.xavier_initializer())
# 		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))

		

# 		self.W_e = tf.get_variable("W_e", shape=[3, 3, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
# 		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
# 		self.centers = tf.get_variable("centers",[1, 1, 1, self.reduction_dim, self.centers_num],
# 			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))



# 		# classification parameters
# 		self.W_c = tf.get_variable("W_c",[self.output_dim,self.voc_size],
# 			initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.output_dim)))
# 		self.b_c = tf.get_variable("b_c",initializer = tf.random_normal([self.voc_size],stddev=1./math.sqrt(self.voc_size)))



# 	def init_embedding_matrix(self):
# 		'''init word embedding matrix
# 		'''
# 		voc_size = self.voc_size
# 		d_w2v = self.d_w2v	
# 		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
# 		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

# 		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
# 		for v in range(voc_size):
# 			LUT[v] = rng.randn(d_w2v)
# 			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

# 		# word 0 is blanked out, word 1 is 'UNK'
# 		LUT[0] = np.zeros((d_w2v))
# 		# setup LUT!
# 		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)

# 		return T_w2v, T_mask 
# 	def encoder(self):
		
# 		timesteps = self.enc_in_shape[1]

# 		input_feature = self.input_feature
# 		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
# 		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
# 		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
# 		input_feature = tf.nn.relu(input_feature)

# 		self.enc_in_shape = input_feature.get_shape().as_list()

		
# 		assignment = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
# 		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, self.centers_num]))
# 		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])
# 		assignment = tf.nn.softmax(assignment,dim=-1)

# 		# for alpha * c
# 		a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
# 		a = tf.multiply(a_sum,self.centers)

# 		# for alpha * x
# 		assignment = tf.transpose(assignment,perm=[0,2,1])

# 		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.enc_in_shape[4]])

# 		vlad = tf.matmul(assignment,input_feature)
# 		vlad = tf.transpose(vlad, perm=[0,2,1])

# 		# for differnce
# 		vlad = tf.subtract(vlad,a)

# 		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1],self.centers_num])
# 		vlad = tf.nn.l2_normalize(vlad,1)

# 		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1]*self.centers_num])
# 		vlad = tf.nn.l2_normalize(vlad,2)

# 		return vlad

	
# 	def build_model(self):
# 		print('building seq model ... ...')
# 		self.init_parameters()
# 		vlad = self.encoder()
# 		return vlad



# class SeqVladAttentionModel(NetVladAttentionModel):
# 	'''
# 		caption model for ablation studying
# 		output_dim = num_of_filter
# 	'''
# 	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, 
# 		centers_num=16,
# 		filter_size=1, stride=[1,1,1,1], pad='SAME', 
# 		done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5,
# 		attention_dim = 100, dropout=0.5,
# 		inner_activation='hard_sigmoid',activation='tanh',
# 		return_sequences=True):

# 		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

# 		self.input_captions = input_captions

# 		self.voc_size = voc_size
# 		self.d_w2v = d_w2v

# 		self.output_dim = output_dim
# 		self.filter_size = filter_size
# 		self.stride = stride
# 		self.pad = pad

# 		self.centers_num = centers_num
# 		# self.init_w = init_w
# 		# self.init_b = init_b
# 		# self.init_centers = init_centers

# 		self.beam_size = beam_size

# 		assert(beamsearch_batchsize==1)
# 		self.batch_size = beamsearch_batchsize
# 		self.done_token = done_token
# 		self.max_len = max_len

# 		self.dropout = dropout



# 		self.inner_activation = inner_activation
# 		self.activation = activation
# 		self.return_sequences = return_sequences
# 		self.attention_dim = attention_dim

# 		self.enc_in_shape = self.input_feature.get_shape().as_list()
# 		self.decoder_input_shape = self.input_captions.get_shape().as_list()
# 		print('enc_in_shape', self.enc_in_shape)

# 		print('activation',self.activation)
# 	def init_parameters(self):
# 		print('init_parameters ...')

# 		# encoder parameters
# 		# print(self.enc_in_shape)
		
# 		self.W_e = tf.get_variable("W_e", shape=[3, 3, self.enc_in_shape[-1], self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
# 		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
# 		self.centers = tf.get_variable("centers",[1, 1, 1, self.enc_in_shape[-1], self.centers_num],
# 			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

# 		# self.init_centers = tf.cast(tf.reshape(tf.transpose(self.init_centers,perm=[1,0]),[-1, self.enc_in_shape[-1], self.centers_num]),tf.float32)
# 		# self.centers = tf.Variable(self.init_centers,dtype=tf.float32, name='centers')

# 		# encoder_i2h_shape = [1, 1, self.enc_in_shape[-1], self.centers_num]		
# 		# self.init_w = tf.cast(tf.reshape(tf.transpose(self.init_w,perm=[1,0]),encoder_i2h_shape),tf.float32)
# 		# self.W_e = tf.Variable(self.init_w,dtype=tf.float32, name='W_e')
# 		# self.b_e = tf.Variable(tf.cast(self.init_b,tf.float32),dtype=tf.float32, name='b_e')


# 		tf.summary.histogram('centers',self.centers)
# 		tf.summary.histogram('W_e',self.W_e)
# 		tf.summary.histogram('b_e',self.b_e)


# 		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
# 		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
# 		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
# 		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 


# 		self.liner_W = tf.get_variable("liner_W",[self.enc_in_shape[-1], self.output_dim],
# 			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

# 		self.liner_b = tf.get_variable("liner_b",initializer=tf.random_normal([self.output_dim],stddev=1./math.sqrt(self.output_dim)))

# 		# decoder parameters
# 		self.T_w2v, self.T_mask = self.init_embedding_matrix()

# 		decoder_i2h_shape = (self.d_w2v,3*self.output_dim)
# 		decoder_h2h_shape = (self.output_dim,self.output_dim)

# 		self.W_d = tf.get_variable("W_d",decoder_i2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.d_w2v)))
# 		self.b_d = tf.get_variable("b_d",initializer = tf.random_normal([3*self.output_dim], stddev=1./math.sqrt(3*self.output_dim)))
		
# 		self.U_d_r = tf.get_variable("U_d_r",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
# 		self.U_d_z = tf.get_variable("U_d_z",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
# 		self.U_d_h = tf.get_variable("U_d_h",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))

		
		

		
# 		self.W_a = tf.get_variable("W_a",[self.enc_in_shape[-1]*self.centers_num,self.attention_dim],
# 			initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.enc_in_shape[-1]*self.centers_num)))

# 		self.U_a = tf.get_variable("U_a",[self.output_dim,self.attention_dim],initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
# 		self.b_a = tf.get_variable("b_a",initializer = tf.random_normal([self.attention_dim],stddev=1. / math.sqrt(self.attention_dim)))

# 		self.W = tf.get_variable("W",(self.attention_dim,1),initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.attention_dim)))

# 		self.A = tf.get_variable("A",(self.enc_in_shape[-1]*self.centers_num,3*self.output_dim),
# 			initializer=tf.random_normal_initializer(stddev=1./ math.sqrt(self.enc_in_shape[-1]*self.centers_num)))

		


# 		# classification parameters
# 		self.W_c = tf.get_variable("W_c",[self.output_dim,self.voc_size],
# 			initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.output_dim)))
# 		self.b_c = tf.get_variable("b_c",initializer = tf.random_normal([self.voc_size],stddev=1./math.sqrt(self.voc_size)))

# 		tf.summary.histogram('W_c',self.W_c)
# 		tf.summary.histogram('b_c',self.b_c)

# 	def init_embedding_matrix(self):
# 		'''init word embedding matrix
# 		'''
# 		voc_size = self.voc_size
# 		d_w2v = self.d_w2v	
# 		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
# 		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

# 		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
# 		for v in range(voc_size):
# 			LUT[v] = rng.randn(d_w2v)
# 			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

# 		# word 0 is blanked out, word 1 is 'UNK'
# 		LUT[0] = np.zeros((d_w2v))
# 		# setup LUT!
# 		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)

# 		return T_w2v, T_mask 
# 	def encoder(self):
		
# 		timesteps = self.enc_in_shape[1]
# 		embedded_feature = self.input_feature
		
# 		assignment = tf.reshape(embedded_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
# 		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, self.centers_num]))
		
# 		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.centers_num])

# 		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
# 		assignment = tf.transpose(assignment, perm=axis)

# 		input_assignment = tf.TensorArray(
# 	            dtype=embedded_feature.dtype,
# 	            size=timesteps,
# 	            tensor_array_name='input_assignment')
# 		if hasattr(input_assignment, 'unstack'):
# 			input_assignment = input_assignment.unstack(assignment)
# 		else:
# 			input_assignment = input_assignment.unpack(assignment)	

# 		hidden_states = tf.TensorArray(
# 	            dtype=tf.float32,
# 	            size=timesteps,
# 	            tensor_array_name='hidden_states')

# 		def get_init_state(x, output_dims):
# 			initial_state = tf.zeros_like(x)
# 			initial_state = tf.reduce_sum(initial_state,axis=[1,4])
# 			initial_state = tf.expand_dims(initial_state,dim=-1)
# 			initial_state = tf.tile(initial_state,[1,1,1,output_dims])
# 			return initial_state
# 		def step(time, hidden_states, h_tm1):
# 			assign_t = input_assignment.read(time) # batch_size * dim

			
# 			r = hard_sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='uh_r'))
# 			z = hard_sigmoid(assign_t+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='uh_z'))
# 			if self.activation=='tanh':
# 				hh = tf.tanh(assign_t+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))
# 			elif self.activation=='softmax':
# 				hh = tf.nn.softmax(assign_t+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'),dim=-1)
# 			elif self.activation=='relu':
# 				hh = tf.nn.relu(assign_t+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))
# 			elif self.activation=='sigmoid':
# 				hh = hard_sigmoid(assign_t+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))
# 			h = (1-z)*hh + z*h_tm1
			
# 			hidden_states = hidden_states.write(time, h)

# 			return (time+1,hidden_states, h)

# 		time = tf.constant(0, dtype='int32', name='time')
# 		initial_state = get_init_state(embedded_feature,self.centers_num)

# 		feature_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=step,
# 	            loop_vars=(time, hidden_states, initial_state ),
# 	            parallel_iterations=32,
# 	            swap_memory=True)


# 		hidden_states = feature_out[-2]
# 		if hasattr(hidden_states, 'stack'):
# 			assignment = hidden_states.stack()
# 		else:
# 			assignment = hidden_states.pack()

		
		
# 		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
# 		assignment = tf.transpose(assignment, perm=axis)


# 		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])
# 		# assignment = tf.nn.softmax(assignment,dim=-1)

# 		# for alpha * c
# 		a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
# 		a = tf.multiply(a_sum,self.centers)
# 		tf.summary.histogram('a',a)
# 		# for alpha * x
# 		assignment = tf.transpose(assignment,perm=[0,2,1])

# 		embedded_feature = tf.reshape(embedded_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.enc_in_shape[4]])

# 		vlad = tf.matmul(assignment,embedded_feature)
# 		vlad = tf.transpose(vlad, perm=[0,2,1])
# 		tf.summary.histogram('vlad',vlad)

# 		# for differnce
# 		vlad = tf.subtract(vlad,a)

# 		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1],self.centers_num])
# 		vlad = tf.nn.l2_normalize(vlad,1)

# 		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1]*self.centers_num])
# 		vlad = tf.nn.l2_normalize(vlad,2)
# 		last_output = tf.nn.xw_plus_b(tf.reduce_mean(self.input_feature,axis=[1,2,3]),self.liner_W, self.liner_b)

# 		return last_output, vlad


class SeqVladWithReduAttentionModel(object):
	'''
		caption model for ablation studying
		output_dim = num_of_filter
	'''
	def __init__(self, input_feature,
		num_class=51,
		reduction_dim=512,
		centers_num=16,
		dropout=0.5,
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


		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences

		self.enc_in_shape = self.input_feature.get_shape().as_list()

	def init_parameters(self):
		print('init_parameters ...')

		# encoder parameters
		# print(self.enc_in_shape)

		self.redu_W = tf.get_variable("redu_W", shape=[3, 3, self.enc_in_shape[-1], self.reduction_dim], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))

		
		self.W_e = tf.get_variable("W_e", shape=[3, 3, self.reduction_dim, self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([self.centers_num],stddev=1./math.sqrt(self.centers_num)))
		self.centers = tf.get_variable("centers",[1, 1, 1, self.reduction_dim, self.centers_num],
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

		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1],self.centers_num])
		vlad = tf.nn.l2_normalize(vlad,1)

		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1]*self.centers_num])
		vlad = tf.nn.l2_normalize(vlad,2)


		train_output = tf.nn.xw_plus_b(tf.reduce_mean(tf.nn.dropout(vlad,self.dropout),axis=1),self.liner_W, self.liner_b)

		test_output = tf.nn.xw_plus_b(tf.reduce_mean(vlad,axis=1),self.liner_W, self.liner_b)

		return train_output,test_output
	
	def build_model(self):
		print('building seq model ... ...')
		self.init_parameters()
		train_output,test_output = self.encoder()
		return train_output,test_output


# class SeqVladWithReduNotShareAttentionModel(object):
# 	'''
# 		caption model for ablation studying
# 		output_dim = num_of_filter
# 	'''
# 	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, 
# 		reduction_dim=512,
# 		centers_num=16,
# 		filter_size=1, stride=[1,1,1,1], pad='SAME', 
# 		done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5,
# 		attention_dim = 100, dropout=0.5,
# 		inner_activation='hard_sigmoid',activation='tanh',
# 		return_sequences=True):

# 		self.reduction_dim=reduction_dim
		

		

# 		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

# 		self.input_captions = input_captions

# 		self.voc_size = voc_size
# 		self.d_w2v = d_w2v

# 		self.output_dim = output_dim
# 		self.filter_size = filter_size
# 		self.stride = stride
# 		self.pad = pad

# 		self.centers_num = centers_num

# 		self.beam_size = beam_size

# 		assert(beamsearch_batchsize==1)
# 		self.batch_size = beamsearch_batchsize
# 		self.done_token = done_token
# 		self.max_len = max_len

# 		self.dropout = dropout



# 		self.inner_activation = inner_activation
# 		self.activation = activation
# 		self.return_sequences = return_sequences
# 		self.attention_dim = attention_dim



# 		self.enc_in_shape = self.input_feature.get_shape().as_list()
# 		self.decoder_input_shape = self.input_captions.get_shape().as_list()
# 		print('enc_in_shape', self.enc_in_shape)

# 		print('activation',self.activation)
# 	def init_parameters(self):
# 		print('init_parameters ...')

# 		# encoder parameters
# 		# print(self.enc_in_shape)

# 		self.redu_W = tf.get_variable("redu_W", shape=[3, 3, self.enc_in_shape[-1], self.reduction_dim], 
# 										initializer=tf.contrib.layers.xavier_initializer())
# 		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))

		
# 		self.W_e = tf.get_variable("W_e", shape=[3, 3, self.reduction_dim, 3*self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
# 		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([3*self.centers_num],stddev=1./math.sqrt(3*self.centers_num)))
# 		self.centers = tf.get_variable("centers",[1, 1, 1, self.reduction_dim, self.centers_num],
# 			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))



# 		tf.summary.histogram('centers',self.centers)
# 		tf.summary.histogram('W_e',self.W_e)
# 		tf.summary.histogram('b_e',self.b_e)


# 		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
# 		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
# 		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
# 		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

# 		if self.output_dim!=self.enc_in_shape[-1]:
# 			print('the dimension of input feature != hidden size')
# 			self.liner_W = tf.get_variable("liner_W",[self.enc_in_shape[-1], self.output_dim],
# 				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

# 			self.liner_b = tf.get_variable("liner_b",initializer=tf.random_normal([self.output_dim],stddev=1./math.sqrt(self.output_dim)))

# 		# decoder parameters
# 		self.T_w2v, self.T_mask = self.init_embedding_matrix()

# 		decoder_i2h_shape = (self.d_w2v,3*self.output_dim)
# 		decoder_h2h_shape = (self.output_dim,self.output_dim)

# 		self.W_d = tf.get_variable("W_d",decoder_i2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.d_w2v)))
# 		self.b_d = tf.get_variable("b_d",initializer = tf.random_normal([3*self.output_dim], stddev=1./math.sqrt(3*self.output_dim)))
		
# 		self.U_d_r = tf.get_variable("U_d_r",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
# 		self.U_d_z = tf.get_variable("U_d_z",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
# 		self.U_d_h = tf.get_variable("U_d_h",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))

		
		

		
# 		self.W_a = tf.get_variable("W_a",[self.reduction_dim*self.centers_num,self.attention_dim],
# 			initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.reduction_dim*self.centers_num)))

# 		self.U_a = tf.get_variable("U_a",[self.output_dim,self.attention_dim],initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
# 		self.b_a = tf.get_variable("b_a",initializer = tf.random_normal([self.attention_dim],stddev=1. / math.sqrt(self.attention_dim)))

# 		self.W = tf.get_variable("W",(self.attention_dim,1),initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.attention_dim)))

# 		self.A = tf.get_variable("A",(self.reduction_dim*self.centers_num,3*self.output_dim),
# 			initializer=tf.random_normal_initializer(stddev=1./ math.sqrt(self.reduction_dim*self.centers_num)))

		


# 		# classification parameters
# 		self.W_c = tf.get_variable("W_c",[self.output_dim,self.voc_size],
# 			initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.output_dim)))
# 		self.b_c = tf.get_variable("b_c",initializer = tf.random_normal([self.voc_size],stddev=1./math.sqrt(self.voc_size)))

# 		tf.summary.histogram('W_c',self.W_c)
# 		tf.summary.histogram('b_c',self.b_c)

# 	def init_embedding_matrix(self):
# 		'''init word embedding matrix
# 		'''
# 		voc_size = self.voc_size
# 		d_w2v = self.d_w2v	
# 		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
# 		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')

# 		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
# 		for v in range(voc_size):
# 			LUT[v] = rng.randn(d_w2v)
# 			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

# 		# word 0 is blanked out, word 1 is 'UNK'
# 		LUT[0] = np.zeros((d_w2v))
# 		# setup LUT!
# 		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)

# 		return T_w2v, T_mask 
# 	def encoder(self):
		
# 		timesteps = self.enc_in_shape[1]
# 		# # reduction
# 		input_feature = self.input_feature
# 		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
# 		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
# 		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
# 		input_feature = tf.nn.relu(input_feature)

# 		self.enc_in_shape = input_feature.get_shape().as_list()

# 		assignment = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
# 		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, 3*self.centers_num]))
		
# 		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],3*self.centers_num])



# 		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
# 		assignment = tf.transpose(assignment, perm=axis)

# 		input_assignment = tf.TensorArray(
# 	            dtype=assignment.dtype,
# 	            size=timesteps,
# 	            tensor_array_name='input_assignment')
# 		if hasattr(input_assignment, 'unstack'):
# 			input_assignment = input_assignment.unstack(assignment)
# 		else:
# 			input_assignment = input_assignment.unpack(assignment)	

# 		hidden_states = tf.TensorArray(
# 	            dtype=tf.float32,
# 	            size=timesteps,
# 	            tensor_array_name='hidden_states')

# 		def get_init_state(x, output_dims):
# 			initial_state = tf.zeros_like(x)
# 			initial_state = tf.reduce_sum(initial_state,axis=[1,4])
# 			initial_state = tf.expand_dims(initial_state,dim=-1)
# 			initial_state = tf.tile(initial_state,[1,1,1,output_dims])
# 			return initial_state
# 		def step(time, hidden_states, h_tm1):
# 			assign_t = input_assignment.read(time) # batch_size * dim
# 			assign_t_r = assign_t[:,:,:,0:self.centers_num]
# 			assign_t_z = assign_t[:,:,:,self.centers_num:2*self.centers_num]
# 			assign_t_h = assign_t[:,:,:,2*self.centers_num::]
			
# 			r = hard_sigmoid(assign_t_r+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='r'))
# 			z = hard_sigmoid(assign_t_z+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='z'))

# 			hh = tf.tanh(assign_t_h+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='hh'))

# 			h = (1-z)*hh + z*h_tm1
			
# 			hidden_states = hidden_states.write(time, h)

# 			return (time+1,hidden_states, h)

# 		time = tf.constant(0, dtype='int32', name='time')
# 		initial_state = get_init_state(input_feature,self.centers_num)

# 		feature_out = tf.while_loop(
# 	            cond=lambda time, *_: time < timesteps,
# 	            body=step,
# 	            loop_vars=(time, hidden_states, initial_state ),
# 	            parallel_iterations=32,
# 	            swap_memory=True)


# 		hidden_states = feature_out[-2]
# 		if hasattr(hidden_states, 'stack'):
# 			assignment = hidden_states.stack()
# 		else:
# 			assignment = hidden_states.pack()

		
		
# 		axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
# 		assignment = tf.transpose(assignment, perm=axis)


# 		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])

# 		# for alpha * c
# 		a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
# 		a = tf.multiply(a_sum,self.centers)
# 		tf.summary.histogram('a',a)
# 		# for alpha * x
# 		assignment = tf.transpose(assignment,perm=[0,2,1])

# 		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])

# 		vlad = tf.matmul(assignment,input_feature)
# 		vlad = tf.transpose(vlad, perm=[0,2,1])
# 		tf.summary.histogram('vlad',vlad)
# 		# for differnce
# 		vlad = tf.subtract(vlad,a)

# 		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1],self.centers_num])
# 		vlad = tf.nn.l2_normalize(vlad,1)

# 		vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1]*self.centers_num])
# 		vlad = tf.nn.l2_normalize(vlad,2)
# 		last_output = tf.reduce_mean(self.input_feature,axis=[1,2,3])
# 		if self.output_dim!=self.input_feature.get_shape().as_list()[-1]:
# 			print('the dimension of input feature != hidden size')
# 			last_output = tf.nn.xw_plus_b(last_output,self.liner_W, self.liner_b)

# 		return last_output, vlad
