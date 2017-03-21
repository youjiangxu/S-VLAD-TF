
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
video_fea_root1 = '/mnt/data/ucf_101/ucf-split'+str(split)+'-resize256x340-5crop224-pool5-feature-no-compress.h5'

video_fea_root2 = '/home/xyj/usr/local/data/ucf-101/rgb-feature/ucf-split'+str(split)+'-crop224-feature.h5'

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



lr=0.05
dropout=0.9
batch_size = 64
test_batch_size = 64
total_train_v_num = len(train_v_list)
total_test_v_num = len(test_v_list)

train_batch_num = total_train_v_num//batch_size
test_batch_num = total_test_v_num//test_batch_size
#-----------------------------------------------------------------------------
iteration_num = 200
test_iter = 1

train_v_list = np.random.permutation(train_v_list)
used_kmeans_list = train_v_list[0:1000]



'''load centers...'''
hf1 = h5py.File(video_fea_root1, 'r')

hf2 = h5py.File(video_fea_root2, 'r')
	
center_file = '/home/xyj/usr/local/data/ucf-101/centers/action_rgb_resize256_crop224_split'+str(split)+'_'+'d'+str(image_filter)+'_k'+str(K_centers)+'.pkl'
(init_w,init_b,init_centers) = CenterUtil.get_action_centers(center_file, hf1, used_kmeans_list, image_filter, K_centers,split=split)

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

'''configure'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config.log_device_placement=False

sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)
K.set_session(sess)











evaluate_step = True


# acc_value = accuracy(labels, predict)
# with sess.as_default():
# 	input_videos = np.random.random((2,5,512,7,7))
# 	input_label = np.random.random((2,101))
# 	print acc_value.eval(feed_dict={seq_img_in: input_videos,
#                                     labels: input_label,
#                                     K.learning_phase(): 0})
print('build done...')

export_path = '/home/xyj/usr/local/saved_model/'+'action_rgb_split'+str(split)+'_10crop/K'+str(K_centers)+'_t'+str(max_timesteps)+'_b'+str(batch_size)+'_lr'+str(lr)+'_d'+str(dropout)+'_l2norm'+str(L2_normalize) # where to save the exported graph

if not os.path.exists(export_path):
	os.makedirs(export_path)
print('train model....')

with sess.as_default():
	saver = tf.train.Saver(sharded=True,max_to_keep=50)
	saver.restore(sess, export_path+'/iter113_L1.12049588864_A0.751643375464.ckpt')

	if(not evaluate_step):
		print('train phase .... total train sample:%d' %(total_train_v_num))
		for iter_idx in xrange(iteration_num):
			'''train'''
			train_v_list = np.random.permutation(train_v_list)
			for batch_idx in xrange(train_batch_num):

				used_train_list = train_v_list[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_v_num)]
				start = time.clock()
				input_videos = DataUtil.get_video_conv_feature_10crop(hf1,hf2,used_train_list)

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

					input_videos = DataUtil.get_video_conv_feature_10crop(hf1,hf2,used_test_list)
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

		for batch_idx, sample in enumerate(test_v_list):
			used_test_list = [sample]

			input_videos1 = DataUtil.get_25_test_data_5crop(hf1,used_test_list)
			input_videos2 = DataUtil.get_25_test_data_5crop(hf2,used_test_list)

			input_label = DataUtil.load_batch_labels(used_test_list,nb_classes,label_dict)

			predict_score1 = sess.run([predict],feed_dict={seq_img_in: input_videos1,labels: input_label,
				K.learning_phase(): 0})

			predict_score2 = sess.run([predict],feed_dict={seq_img_in: input_videos2,labels: input_label,
				K.learning_phase(): 0})
			
			
			final_score = np.average(predict_score1[0],axis=0)+np.average(predict_score2[0],axis=0)
			predict_video = np.argmax(final_score)


			output_root = '/home/xyj/usr/local/predict_result/rgb_split'+str(split)+'/K'+str(K_centers)+'_d'+str(dropout)+'_lr'+str(lr)+'iter113_L1.1204_A0.7516'
			temp = used_test_list[0].split('/')
			if not os.path.exists(output_root+'/'+temp[0]):
				os.makedirs(output_root+'/'+temp[0])

			output = open(output_root+'/'+temp[0]+'/'+temp[1][0:-4]+'.pkl', 'wb')
			cPickle.dump(final_score,output,protocol=2)

			output.close()



			ground_truth = np.argmax(input_label)
			if(predict_video==ground_truth):
				ave_acc+=1
			print('predict class:%d, ground class:%d, used_frame:%d, right video num:%d/%d' %(predict_video, ground_truth,len(input_videos1)*2, ave_acc, (batch_idx+1)))

			

			# total_loss = total_loss+hist[0]
			# ave_acc = ave_acc+hist[1]*test_batch_size
		# total_loss = ave_acc/test_batch_num
		ave_acc = ave_acc /total_test_v_num
		print('test phase--------acc:%.5f' %(ave_acc))