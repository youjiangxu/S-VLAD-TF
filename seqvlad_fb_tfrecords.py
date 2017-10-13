import numpy as np
import os
import h5py
import math

from utils import DataUtil
from model import SeqVladFBModel 

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import tensorflow as tf
import cPickle as pickle
import time
import json

import argparse
	
def decode_from_tfrecords(tfrecords_file, is_batch):
	reader = tf.TFRecordReader()
	_, serialized_example =reader.read(tfrecords_file) # return file name and file
	features = tf.parse_single_example(serialized_example, features={
			'label': tf.FixedLenFeature([], tf.int64),
			'feature': tf.FixedLenFeature([], tf.string),
		})
	feature = tf.decode_raw(features['feature'], tf.float32)
	feature = tf.reshape(feature, [10, 1024, 7 ,7])
	label = tf.cast(features['label'], tf.int64)

	isFlip = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None)
	feature = tf.cond(isFlip>0, lambda: tf.reverse(feature,axis=[0]), lambda: feature)

	if is_batch:
		batch_size = 128
		min_after_dequeue = 256
		num_threads = 8
		capatity = min_after_dequeue + num_threads*batch_size
		feature, label = tf.train.shuffle_batch([feature,label],
			batch_size = batch_size,
			num_threads = num_threads,
			capacity = capatity,
			min_after_dequeue = min_after_dequeue,
			)

	return feature, label

def exe_train(sess, data, epoch, batch_size,  feature_shape, 
	train, loss, input_video, y,
	bidirectional=False, step=False,modality='rgb' ):

	total_data = len(data)
	num_batch = int(math.ceil(total_data/batch_size))+1

	
	total_loss = 0.0

	
		# while not coord.should_shop()
		
	for batch_idx in xrange(num_batch):

		# batch_data = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		# tic = time.time()		
		# data_v,data_y = DataUtil.getOversamplingBatchVideoFeature(batch_data,hf,(10,feature_shape[1],feature_shape[2],feature_shape[3]),modality=modality)

		# if bidirectional:
		# 	flag = np.random.randint(0,2)
		# 	if flag==1:
		# 		data_v = data_v[:,::-1]
		# data_time = time.time()-tic
		tic = time.time()
		# print('data_v mean:', np.mean(data_v),' std:', np.std(data_v))
		_, l, labels = sess.run([train,loss,y])
		# print labels
		run_time = time.time()-tic
		total_loss += l
		print('    batch_idx:%d/%d, loss:%.5f, run_time:%.3f' %(batch_idx+1,num_batch,l, run_time))
	total_loss = total_loss/num_batch
	


	return total_loss



def test_model(sess, data, hf, batch_size,  
	loss, predicts, input_video, y, test_output, ):
	# if feature=='google':
	# 	hf = h5py.File('/mnt/data3/xyj/data/ucf101/feature/test_split_'+str(split)+'_in5b_'+str(modality)+'_10crop.h5','r')
	# elif feature=='vgg':
	# 	hf = h5py.File('/mnt/data3/xyj/data/ucf101/feature/vgg_test_split_'+str(split)+'_pool5_'+str(modality)+'_10crop.h5','r')

	total_acc = 0.0
	total_data = len(data)
	scores = []
	labels = []
	for idx, samples in enumerate(data):
		data_v=[]
		for x in xrange(10):
			data_v.append(hf[samples[0]+'/'+str(x)])
		data_y = [samples[1]]
		[gw] = sess.run([predicts],feed_dict={input_video:np.asarray(data_v),y:data_y})
		

		scores.append(gw)
		labels.append(samples[1])

		cur = np.argmax(np.mean(gw,axis=0),axis=-1)
		batch_acc = np.sum(np.where(cur==data_y,1,0))
		total_acc+= batch_acc
		print('    vid:%d, cur:%d, gt:%d, acc:%d/%d' %(idx+1,cur,data_y[0],total_acc,total_data))
		
	total_acc=total_acc*1.0/total_data
	video_scores = [scores,labels]
	if test_output is not None:
		np.savez(test_output, scores=video_scores, labels=labels)
	return total_acc


def main(hf2,f_type,
		redu_filter_size = 3,
		feature='google',
		model='seqvlad',
		modality='rgb',
		test=False,
		test_output=None,
		split=1,
		step=False,
		dropout=0.5,
		bidirectional=False,
		reduction_dim=512,
		activation = 'tanh',
		centers_num = 32, kernel_size=1, d_w2v=512, 
		feature_shape=None,lr=0.01,
		batch_size=64,total_epoch=100,
		file=None,pretrained_model=None):
	

	# Create vocabulary
	train_data, test_data = DataUtil.get_data(split=int(split),file=file)

	print('building model ...')
	

	if test:
		input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
		y = tf.placeholder(tf.int32,shape=(None,))
		
	else:
		output_path = '/mnt/data3/xyj/data/hmdb51/tfrecord_feature/split'+str(split)+'_'+str(modality)+'_train.tfrecord'
		tfrecords_file_queue = tf.train.string_input_producer([output_path], num_epochs=None)
		input_video, y = decode_from_tfrecords(tfrecords_file_queue, is_batch=True)
		

	if model=='seqvlad_fb_v1':
		actionModel = SeqVladFBModel.SeqVladFBModel_v1(input_video,
									num_class=51,
									redu_filter_size = args.redu_filter_size,
									dropout=dropout,
									reduction_dim=reduction_dim,
									activation=activation,
									centers_num=centers_num, 
									filter_size=kernel_size)
	elif model=='seqvlad_fb_v2':
		actionModel = SeqVladFBModel.SeqVladFBModel_v2(input_video,
									num_class=51,
									redu_filter_size = args.redu_filter_size,
									dropout=dropout,
									reduction_dim=reduction_dim,
									activation=activation,
									centers_num=centers_num, 
									filter_size=kernel_size)
	elif model=='seqvlad_fb_v3':
		actionModel = SeqVladFBModel.SeqVladFBModel_v3(input_video,
									num_class=51,
									redu_filter_size = args.redu_filter_size,
									dropout=dropout,
									reduction_dim=reduction_dim,
									activation=activation,
									centers_num=centers_num, 
									filter_size=kernel_size)
	
	else:
		# assert model in ['seqvlad','netvlad','notshare']
		assert model in ['seqvlad_fb_v1','seqvlad_fb_v2','seqvlad_fb_v3']

		exit()
		
	train_predicts, test_predicts = actionModel.build_model()
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=train_predicts)

	print('regularized parameters:')
	vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	for var in vars:
		print var.name
	loss = tf.reduce_mean(loss) #+ 0.0001*tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

	optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr,name='sgd')
	

	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_global_norm([grad], 10)[0][0], var) for grad, var in gvs ]
	train = optimizer.apply_gradients(capped_gvs)




	'''
		configure && runtime environment
	'''
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	config.log_device_placement=False

	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)
	
	'''
		tensorboard configure
	'''
	export_path = '/home/xyj/usr/local/saved_model/'+str(model)+'_hmdb51/'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])+'_B'+str(batch_size)

	with sess.as_default():
		saver = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
		if pretrained_model is not None:
			saver.restore(sess, pretrained_model)
			print('restore pre trained file:' + pretrained_model)


		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			if test:
				tic = time.time()
				total_acc = test_model(sess, test_data, hf2, batch_size,
											loss, test_predicts, input_video, y, test_output, )
				print('    --Test--, .......Time:%.3f, total_acc:%.5f' %(time.time()-tic,total_acc))
			else:

				
				for epoch in xrange(total_epoch):
					# # shuffle
					print('Epoch: %d/%d, Batch_size: %d' %(epoch+1,total_epoch,batch_size))
					# # train phase
					tic = time.time()
					total_loss = exe_train(sess, train_data, epoch, batch_size, feature_shape, train, loss, input_video, y, 
						 bidirectional=bidirectional, step=step, modality=modality)

					print('    --Train--, Loss: %.5f, .......Time:%.3f' %(total_loss,time.time()-tic))

					# tic = time.time()
					# total_acc, test_loss = exe_test(sess, test_data, batch_size, hf2, feature_shape, 
					# 							loss, test_predicts, input_video, y, modality=modality)
					# print('    --Test--, .......Time:%.3f, total_acc:%.5f, test_loss:%.5f' %(time.time()-tic,total_acc,test_loss))


					

					#save model
					
					if not os.path.exists(export_path+'/model'):
						os.makedirs(export_path+'/model')
						print('mkdir %s' %export_path+'/model')

					save_path = saver.save(sess, export_path+'/model/'+'E'+str(epoch+1)+'_TaL'+str(total_loss)+'.ckpt')
					print("Model saved in file: %s" % save_path)
		except tf.errors.OutOfRangeError:
			print('Done reading')
		finally:
			coord.request_stop()

		coord.request_stop()
		coord.join(threads)
		
def parseArguments():
	parser = argparse.ArgumentParser(description='seqvlad, youtube, video captioning, reduction app')
	
	parser.add_argument('--dataset',type=str,default='hmdb51', choices=['ucf101', 'hmdb51'],
							help='ucf101 or hmdb51')

	parser.add_argument('--feature', type=str, default='google',
							help='google or vgg')

	parser.add_argument('--bidirectional', action='store_true',
							help='bidirectional training')

	parser.add_argument('--step', action='store_true',
							help='step training')

	parser.add_argument('--gpu_id', type=str, default="0",
							help='specify gpu id')

	parser.add_argument('--lr', type=float, default=0.0001,
							help='learning reate')
	parser.add_argument('--epoch', type=int, default=20,
							help='total runing epoch')
	parser.add_argument('--centers_num', type=int, default=16,
							help='the number of centers')
	parser.add_argument('--reduction_dim', type=int, default=256,
							help='the reduction dim of input feature, e.g., 1024->512')

	parser.add_argument('--redu_filter_size', type=int, default=3,
							help='the size of reduction parameters, e.g., 3')

	parser.add_argument('--dropout', type=float, default=0.5,
							help='the keep probability ')

	parser.add_argument('--test', action='store_true',
							help='testing model')

	parser.add_argument('--test_output', type=str, default=None,
							help='the path where the test output to save')

	parser.add_argument('--pretrained_model', type=str, default=None,
							help='the pretrained model')

	parser.add_argument('--modality', type=str, default='rgb',
							help='rgb or flow')

	parser.add_argument('--split', type=str, default=1,
							help='the split which to train or test')

	parser.add_argument('--model',type=str,default='seqvlad_fb_v1',
							help='netvlad, seqvlad, or notshare')



	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parseArguments()
	print(args)
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
	lr = args.lr
	epoch = args.epoch


	reduction_dim=args.reduction_dim
	centers_num = args.centers_num
	bidirectional = args.bidirectional
	step = args.step
	dropout = args.dropout
	feature = args.feature

	kernel_size = 3
	
	activation = 'tanh' ## can be one of 'tanh,softmax,relu,sigmoid'

	test = args.test

	split = args.split

	assert args.model in ['seqvlad_fb_v1','seqvlad_fb_v2','seqvlad_fb_v3']
	assert args.dataset in ['hmdb51','ucf101']
	if feature=='google':
		video_feature_dims = 1024
		timesteps_v = 10 # sequences length for video
		height = 7
		width = 7
		feature_shape = (timesteps_v,video_feature_dims,height,width)
		f_type = args.dataset+'_split'+str(split)+'_'+args.model \
		+'_'+feature+'_k'+str(kernel_size)+'_c'+str(centers_num) \
		+'_redu'+str(reduction_dim)+'_d'+str(dropout)+'_'+str(args.modality)+'_rfs'+str(args.redu_filter_size)
		# feature_path = '/home/xyj/usr/local/data/mvad/feature/mvad-google-in5b-'+str(timesteps_v)+'fpv.h5'
	
	else:
		assert feature in ['google']
		exit()
	
	if step:
		f_type = 'step_'+ f_type
	if bidirectional:
		f_type = 'bi_'+ f_type
	# f_type = 'soft_'+f_type
	# feature_path = '/data/xyj/resnet152_pool5_f'+str(timesteps_v)+'.h5'
	# feature_path = '/home/xyj/usr/local/data/youtube/in5b-'+str(timesteps_v)+'fpv.h5'
	modality = args.modality
	assert modality in ['rgb','flow']
	gt_file = '/mnt/data3/xyj/data/'+args.dataset+'/gt/'+args.dataset+'.json'
	
	# feature_path1 = '/mnt/data3/xyj/data/'+args.dataset+'/feature/train_split_'+str(split)+'_in5b_'+str(modality)+'_10crop.h5'
	feature_path2 = '/mnt/data3/xyj/data/'+args.dataset+'/feature/test_split_'+str(split)+'_in5b_'+str(modality)+'_10crop.h5'
	
	# hf1 = h5py.File(feature_path1,'r')
	hf2 = h5py.File(feature_path2,'r')

	pretrained_model = args.pretrained_model
	test_output = args.test_output
	if test:
		assert pretrained_model is not None
		assert test_output is not None
		
		
	
	main(hf2,f_type,
		feature=feature,
		redu_filter_size = args.redu_filter_size,
		model=args.model,
		split=split,
		modality=args.modality,
		test=test,
		test_output=test_output,
		step=step, 
		dropout=dropout,
		bidirectional=bidirectional,
		reduction_dim=reduction_dim,
		activation=activation,
		centers_num=centers_num, kernel_size=kernel_size,
		feature_shape=feature_shape,lr=lr,
		batch_size=128,total_epoch=epoch,
		file=gt_file,pretrained_model=pretrained_model)
	

	
	
	
	


	
