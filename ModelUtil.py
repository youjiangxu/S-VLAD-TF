from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, merge, Embedding
from keras.layers import Flatten, Dropout,AveragePooling2D, TimeDistributed,GRU
from keras.layers.core import Activation, Lambda, Reshape
from keras.optimizers import Adam,RMSprop,Adadelta, SGD
from keras.layers import GRU, SeqVLADAction

from keras.utils.visualize_util import plot
from six.moves import cPickle
from keras import backend as K
import theano, sys, json, time, os
import numpy as np

import tensorflow as tf

# import CenterUtil
# import caption_datathing
# import test_caption_model
from eval import eval


def get_action_model(init_w,init_b,init_centers,model_file=None):
	lr = 0.001
	dropout = 0.5
	max_timesteps=10
	image_filter=512
	image_width=7
	image_height=7
	nb_classes=101
	K_centers=16
	reduction_dim=512

	sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=False)
	print('lr: %f; dropout: %f' %(lr,dropout))
	print('max_timesteps:%d, image_filter:%d, image_width:%d, image_height:%d, nb_classes:%d' %(max_timesteps,image_filter,image_width,image_height,nb_classes))
	''' how to load model'''

	print('build model')
	seq_img_in = tf.placeholder(tf.float32, shape=(None,max_timesteps,image_filter, image_height, image_width))
	# seq_img_in = Input(shape=(max_timesteps,image_filter, image_height, image_width))

	netVLAD = SeqVLADAction(K_centers,reduction_dim,init_w,init_b,init_centers)(seq_img_in)
	flatten_layer = Flatten()(netVLAD)

	dropout_layer = Dropout(dropout)(flatten_layer)
	dense = Dense(nb_classes,activation='softmax')(dropout_layer)

	return (seq_img_in,dense)
	# rcn_model = Model(seq_img_in, dense)

	# print('compiling model....')
	# # rcn_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'] )


	
	# if model_file is not None:

	# 	print('loading weight file : ' + str(model_file))
	# 	rcn_model.load_weights(model_file)

	# return rcn_model







if __name__=='__main__':
	# os.makedirs(model_file)
	dict_size = 10
	train_data=''
	K_centers = 16	
	center_dim = 512

	center_file = ''
	# (init_w,init_b,init_centers) = CenterUtil.get_crop224_opencv_norm_clsts(center_file, train_datas sample_dim, k_clusters)

	init_w = np.random.random((K_centers,center_dim))
	init_b = np.random.random((K_centers,))
	init_centers = np.random.random((K_centers,center_dim))


	model = getModel(dict_size, K_centers, center_dim, init_w, init_b, init_centers)
	input_v_data = np.random.random((10,10,512,7,7))
	input_w_data = np.random.randint(1,10,size=(10,16))
	input_label = np.random.random((10,16,10))

	model.train_on_batch([input_v_data,input_w_data], input_label, sample_weight=None)



