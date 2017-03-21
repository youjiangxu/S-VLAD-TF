
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from six.moves import cPickle

import theano, sys, json, time, os
import numpy as np
import h5py



import tensorflow as tf
from keras import backend as K

from keras.layers import SeqVLADAction, Dense, Dropout, Flatten, Input
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2

import ModelUtil
import DataUtil
import CenterUtil


print('preparing parameter ...')

# K.set_learning_phase(1)
split = 2

train_split_file = '/home/xyj/usr/local/data/ucf-101/ucfTrainTestlist/trainlist0'+str(split)+'.txt'
test_split_file = '/home/xyj/usr/local/data/ucf-101/ucfTrainTestlist/testlist0'+str(split)+'.txt'
video_fea_root = '/home/xyj/usr/local/data/ucf-101/flow-feature/ucf-split'+str(split)+'-reseize256-crop224-feature-noRelu.h5'

label_file='/home/xyj/usr/local/data/ucf-101/ucfTrainTestlist/classInd.txt'

train_v_list = [] 
test_v_list = []
with open(train_split_file,'r') as reader:
	for line in reader:
		temp = line.strip().split(' ')
		train_v_list.append(temp[0])
with open(test_split_file,'r') as reader:
	for line in reader:
		temp = line.strip().split(' ')
		test_v_list.append(temp[0])

label_dict = {}
with open(label_file,'r') as reader:
	for line in reader:
		temp = line.strip().split(' ')
		label_dict[temp[1]]=int(temp[0])



max_timesteps = 10
image_width = 7
image_height = 7
image_filter = 512
video_shape = (max_timesteps,image_filter,image_height,image_width)

'''l2 normalize'''
L2_normalize=False


K_centers = 16

nb_classes = 101



lr=0.0002
dropout=0.9
batch_size = 64
test_batch_size = 64
total_train_v_num = len(train_v_list)
total_test_v_num = len(test_v_list)

train_batch_num = total_train_v_num//batch_size
test_batch_num = total_test_v_num//test_batch_size
#-----------------------------------------------------------------------------
iteration_num = 100
test_iter = 1

train_v_list = np.random.permutation(train_v_list)
used_kmeans_list = train_v_list[0:1000]



'''load centers...'''
hf = h5py.File(video_fea_root, 'r')

center_file = '/home/xyj/usr/local/data/ucf-101/centers/action_flow_resize256_crop224_split'+str(split)+'_'+'d'+str(image_filter)+'_k'+str(K_centers)+'_noRelu.pkl'
(init_w,init_b,init_centers) = CenterUtil.get_action_centers(center_file, hf, used_kmeans_list, image_filter, K_centers,split=split)

# init_w = np.zeros((16,512,1,1), dtype=np.float32)
# init_b = np.zeros((16,1,1,1), dtype=np.float32)
# init_centers = np.zeros((16,512,1,1), dtype=np.float32)

'''get trained model'''
# with tf.device('/gpu:1'):
# seq_img_in = tf.placeholder(tf.float32, shape=(None, 5, 512, 7, 7))
labels = tf.placeholder(tf.float32, shape=(None,101))

seq_img_in = Input(shape=video_shape)
# seq_img_in = Input(shape=(max_timesteps,image_filter, image_height, image_width))

seq_vlad = SeqVLADAction(K_centers,image_filter,init_w,init_b,init_centers,inner_init='glorot_uniform',L2_normalize=L2_normalize)(seq_img_in)

flatten = Flatten()(seq_vlad)

drop_layer = Dropout(dropout)(flatten)

predict = Dense(101, activation='softmax')(drop_layer)




loss = tf.reduce_mean(categorical_crossentropy(labels, predict))
acc_value = accuracy(labels, predict)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
# train_step = tf.train.RMSPropOptimizer(lr).minimize(loss)

'''configure'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config.log_device_placement=False

sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)
K.set_session(sess)











evaluate_step = False


# acc_value = accuracy(labels, predict)
# with sess.as_default():
# 	input_videos = np.random.random((2,5,512,7,7))
# 	input_label = np.random.random((2,101))
# 	print acc_value.eval(feed_dict={seq_img_in: input_videos,
#                                     labels: input_label,
#                                     K.learning_phase(): 0})
print('build done...')

export_path = '/home/xyj/usr/local/saved_model/'+'action_split'+str(split)+'/K'+str(K_centers)+'_t'+str(max_timesteps)+'_b'+str(batch_size)+'_lr'+str(lr)+'_d'+str(dropout)+'_l2norm'+str(L2_normalize) # where to save the exported graph

if not os.path.exists(export_path):
	os.makedirs(export_path)
print('train model....')

with sess.as_default():
	saver = tf.train.Saver(sharded=True,max_to_keep=iteration_num)
	# saver.restore(sess, export_path+'/iter46_L1.83788919752_A0.672055314937.ckpt')

	if(not evaluate_step):
		print('train phase .... total train sample:%d' %(total_train_v_num))
		for iter_idx in xrange(iteration_num):
			'''train'''
			train_v_list = np.random.permutation(train_v_list)
			for batch_idx in xrange(train_batch_num):

				used_train_list = train_v_list[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_v_num)]
				start = time.clock()
				input_videos = DataUtil.get_video_conv_feature(hf,used_train_list)

				input_label = DataUtil.load_batch_labels(used_train_list,nb_classes,label_dict)

				load_data_time = (time.clock()-start)
				start = time.clock()

				
				_, loss_val, acc_val = sess.run([train_step, loss, acc_value],feed_dict={seq_img_in: input_videos,
					labels: input_label,K.learning_phase(): 1})

				train_time = (time.clock()-start)
				print('train iter:%d, batch_idx:%d/%d, batch_size:%d, loss:%.5f, acc:%.5f, load_data_time:%.5f s, train_time:%.5f s' %(iter_idx+1,batch_idx+1,train_batch_num,batch_size,loss_val,acc_val,load_data_time,train_time))

			if((iter_idx+1)%test_iter==0):
				total_loss = 0.0
				ave_acc = 0.0
				print('test phase .... total test sample:%d' %(total_test_v_num))
				for batch_idx in xrange(test_batch_num+1):
					used_test_list = test_v_list[batch_idx*test_batch_size:min((batch_idx+1)*test_batch_size,total_test_v_num)]

					input_videos = DataUtil.get_video_conv_feature(hf,used_test_list)
					input_label = DataUtil.load_batch_labels(used_test_list,nb_classes,label_dict)

					acc_val, loss_val = sess.run([acc_value,loss], feed_dict={seq_img_in: input_videos,
	                                    labels: input_label,
	                                    K.learning_phase(): 0})
					print('-->>test iter:%d, batch_idx:%d/%d, test_batch_size:%d, loss:%.5f, acc:%.5f' %(iter_idx+1,batch_idx+1,test_batch_num+1,test_batch_size,loss_val,acc_val))
					total_loss = total_loss+loss_val
					ave_acc = ave_acc+acc_val*test_batch_size
				total_loss = total_loss/(test_batch_num+1)
				ave_acc = ave_acc /total_test_v_num
				print('test phase--------total_loss:%.5f acc%.5f' %(total_loss,ave_acc))
				



				
				# model_exporter = exporter.Exporter(saver)
				save_path = saver.save(sess, export_path+'/'+'iter'+str(iter_idx+1)+'_L'+str(total_loss)+'_A'+str(ave_acc)+'.ckpt')
				print("Model saved in file: %s" % save_path)
				# signature = exporter.classification_signature(input_tensor=seq_img_in,
				#                                               scores_tensor=predict)
				# model_exporter.init(sess.graph.as_graph_def(),
				#                     default_graph_signature=signature)
				# model_exporter.export(export_path, tf.constant(export_version), sess)


	# '''test 25 inputs'''

	if(evaluate_step):

		ave_acc = 0.0		
		print('test phase .... total test sample:%d' %(total_test_v_num))
		test_batch_size = 1

		used_resize_feature = True
		if used_resize_feature:
			resezi_video_fea_root = '/home/xyj/usr/local/data/ucf-101/flow-feature/ucf-split'+str(split)+'-resize224-pool5-feature.h5'
			resize_hf = h5py.File(resezi_video_fea_root, 'r')	 # for test

		for batch_idx, sample in enumerate(test_v_list):
			used_test_list = [sample]

			input_videos = DataUtil.get_25_test_data(hf,used_test_list)
			
				

			input_label = DataUtil.load_batch_labels(used_test_list,nb_classes,label_dict)

			# predict_10_crop = rcn_model.predict(input_videos)
			predict_score = sess.run([predict],feed_dict={seq_img_in: input_videos,labels: input_label,
				K.learning_phase(): 0})


			if used_resize_feature:
				resize_input_videos = DataUtil.get_25_test_data(resize_hf,used_test_list,test_sub_volumn=15)
				resize_predict_score = sess.run([predict],feed_dict={seq_img_in: resize_input_videos,labels: input_label,
				K.learning_phase(): 0})
				predict_score = np.average(predict_score[0],axis=0)+np.average(resize_predict_score[0],axis=0)
				predict_video = np.argmax(predict_score)
			else:
				predict_score = np.average(predict_score[0],axis=0)
				predict_video = np.argmax(predict_score)



			
			output_root = '/home/xyj/usr/local/predict_result/flow_split'+str(split)+'/K'+str(K_centers)+'_d'+str(dropout)+'_lr'+str(lr)
			temp = used_test_list[0].split('/')
			if not os.path.exists(output_root+'/'+temp[0]):
				os.makedirs(output_root+'/'+temp[0])

			output = open(output_root+'/'+temp[0]+'/'+temp[1][0:-4]+'.pkl', 'wb')
			cPickle.dump(predict_score,output,protocol=2)
			output.close()


			# print(len(predict_score[0]))
			# print(predict_score[0])
			# temp = used_test_list[0].split('/')
			# output_root = '/home/lyb/XYJ/CVPR16/predict/rgb/K'+str(K_centers)+'_D'+str(reduction_dim)+'_d'+str(dropout)+'_lr'+str(lr)
			# if not os.path.exists(output_root+'/'+temp[0]):
				# os.makedirs(output_root+'/'+temp[0])

			# output = open(output_root+'/'+temp[0]+'/'+temp[1][0:-4]+'.pkl', 'wb')
			# cPickle.dump(np.average(predict_10_crop,axis=0),output,protocol=2)
			# output.close()

			# print(predict_10_crop.shape)
			

			ground_truth = np.argmax(input_label)
			if(predict_video==ground_truth):
				ave_acc+=1
			print('predict class:%d, ground class:%d, used_frame:%d, right video num:%d/%d' %(predict_video, ground_truth,len(input_videos), ave_acc, (batch_idx+1)))

			

			# total_loss = total_loss+hist[0]
			# ave_acc = ave_acc+hist[1]*test_batch_size
		# total_loss = ave_acc/test_batch_num
		ave_acc = ave_acc /total_test_v_num
		print('test phase--------acc:%.5f' %(ave_acc))