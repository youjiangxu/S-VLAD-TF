from six.moves import cPickle
from keras import backend as K
import theano, sys, json, time, os
import numpy as np
import h5py

import ModelUtil
import datathing
import CenterUtil



def evaluate_mode_by_shell(res_path):
	command ='./eval/call_python_caption_eval.sh '+ res_path
	os.system(command)


if __name__=='__main__':
	# os.makedirs(model_file)
	data_path = '/home/xyj/usr/local/data/YouTube/feature/in5b-10fpv.h5'
	hf = h5py.File(data_path,'r')


	timesteps = 10
	

	center_dim = 1024
	k_clusters = 16
	center_file = '/home/xyj/usr/local/data/YouTube/centers/att_nt_fs/'

	# get k-means cluster centers
	print('the shape of init_w,init_b,init_centers:')

	(init_w,init_b,init_centers) = CenterUtil.get_clsts(center_file, hf, center_dim, k_clusters)
	

	
	# print(np.mean(init_w,axis=-1))
	# get vocab
	vocab_path = '/home/lyb/XYJ/YouTube2Text/GT/new_vocab_info.pkl'
	(vocab,vocab_reverse,vocab_stat) = cPickle.load(open(vocab_path, "rb"))
	# print(vocab)
	
	# get model
	used_optimizers = 'rmsprop'
	learning_rate = 0.0002
	dropout = 0.5
	caption_out_dim=512
	# pretrained_model_file='/home/lyb/XYJ/YouTube2Text/saved_model/GAN/t10_lr0.0002_d0.5_B64_cod512/E29_M0.310720565629.h5'
	model = ModelUtil.get_action_model(len(vocab), k_clusters, center_dim, init_w, init_b, init_centers, 
		model_file=None,used_optimizers=used_optimizers,timesteps=timesteps,learning_rate=learning_rate,dropout=dropout,caption_out_dim=caption_out_dim)


	batchsize = 64
	
	caption_path = '/home/lyb/XYJ/YouTube2Text/GT/new_processed_caption.pkl' # caption length large than 4 and small than 16
	(train_list,val_list,test_list) = cPickle.load(open(caption_path, "rb"))
	print('len of train sample: %d' %(len(train_list)))

	total_epoch = 30
	total_train_sample = len(train_list)
	batch_num = total_train_sample//batchsize

	# np.random.seed(47)
	for epoch in xrange(total_epoch):
		train_list = np.random.permutation(train_list)
		for batch in xrange(batch_num):#batch_num
			start = time.clock()

			batch_caption = train_list[batch*batchsize:min((batch+1)*batchsize,len(train_list))]
			input_caption, label = datathing.get_caption(batch_caption,vocab)
			input_vid_feature = datathing.get_video_conv_feature(batch_caption,hf,encoder_length=timesteps)

			# print(np.sum(np.sum(label,axis=-1),axis=-1))

			# (input_seqs,labels) = Youtobe2textModel.convert_label_to_one_hot(input_label,len(vocab))
			hist = model.train_on_batch([input_vid_feature,input_caption], label, sample_weight=None)

			train_time = time.clock() -start
			print('current_epoch:%d, batch_idx:%d/%d, loss:%.5f, acc:%.5f, train_time:%.5f s' %(epoch+1,batch+1,batch_num,hist[0],hist[1],train_time))
			

		js = ModelUtil.test_seqvlad_caption_model(model, hf, batchsize, vocab_reverse, timesteps=timesteps)

		res_root = '/home/lyb/XYJ/YouTube2Text/result/seqVLAD_caption_softattention/'\
			't'+str(timesteps)+'_lr'+str(learning_rate)+'_d'+str(dropout)+'_B'+str(batchsize)+'_cod'+str(caption_out_dim)

		if not os.path.exists(res_root):
			os.makedirs(res_root)
		res_path = res_root+'/E'+str(epoch+1)+'_b'+str(batchsize)+'.json'
		
		with open(res_path, 'w') as f:
			json.dump(js, f)

		evaluate_mode_by_shell(res_path)

		out = json.load(open(res_path + '_out.json','r'))

		model_file = '/home/lyb/XYJ/YouTube2Text/saved_model/seqVLAD_caption_softattention/'\
		+'t'+str(timesteps)+'_lr'+str(learning_rate)+'_d'+str(dropout)+'_B'+str(batchsize)+'_cod'+str(caption_out_dim)
		if not os.path.exists(model_file):
			os.makedirs(model_file)
		model.save_weights(model_file+'/E'+str(epoch)+'_M'+str(out['METEOR'])+'.h5',overwrite=True)

		