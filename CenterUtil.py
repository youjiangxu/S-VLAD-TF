import numpy as np 
import math

from sklearn.cluster import KMeans
from six.moves import cPickle
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from numpy import linalg as LA
import sys, os
import h5py

def get_centers(center_file, hf, used_kmeans_list, sample_dim, k_clusters, input_norm=False, alpha_state=False, output_root=None,task='action', split=None):
	
	


	# if output_root is None:
	# 	output_root = '/home/xyj/usr/local/data/ucf-101/centers'

	if(not os.path.isfile(center_file)):
		# if not os.path.exists(output_root):
		# 	os.makedirs(output_root)
		
		# file = 'd'+str(sample_dim)+'_k'+str(k_clusters)+'.pkl'
		# if split is not None:
		# 	file = 'split'+str(split)+'_'+file
		# if task is not None:
		# 	file = task+'_'+file


		len_train_data = len(used_kmeans_list)

		max_iter = 500
		random_state = 47		
		n_samples = 256*len_train_data
		sampled_descriptor = np.zeros((n_samples,sample_dim))

		
		for idx, sample in enumerate(used_kmeans_list):
			print('%d, vid: %s' %(idx,sample))
			vid_name = sample.strip().split('/')[1][:-4]
			loaded_video_feature = hf[vid_name]
			loaded_video_feature = np.asarray(loaded_video_feature).reshape(-1,sample_dim,7,7)

			loaded_video_feature = np.swapaxes(loaded_video_feature,1,3)
			descriptors = loaded_video_feature.reshape(-1,sample_dim)

			des_num = len(descriptors)
			des_indexs = np.random.randint(0,des_num,size=256)
			sampled_descriptor[idx*256:(idx+1)*256] = descriptors[des_indexs]

		if input_norm:
			square_sum = np.sum(sampled_descriptor**2,axis=-1,keepdims=True)
			sampled_descriptor = sampled_descriptor*1.0/np.sqrt(square_sum+sys.float_info.epsilon)

		kmeans = KMeans(n_clusters=k_clusters, random_state=random_state, n_jobs=2, max_iter=max_iter ).fit(sampled_descriptor)
		centers = kmeans.cluster_centers_
		nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(centers)

		distances, indices = nbrs.kneighbors(sampled_descriptor)
		ave_dis = np.mean(distances[:,1]-distances[:,0])+sys.float_info.epsilon
		
		output = open(center_file, 'wb')
		cPickle.dump((centers,ave_dis),output,protocol=2)
		output.close()
	else:
		f = open(center_file, 'rb')
		(centers,ave_dis) = cPickle.load(f)
		f.close()
	
	if alpha_state:
		alpha = -math.log(0.01)/ave_dis
	else:
		alpha = 1
	print('alpha:%d' %alpha)

	# centers = centers- np.mean(centers,axis=1,keepdims=True)

	init_w = alpha*2*centers
	init_b = -alpha*np.sum(centers**2,axis=1)

	init_centers = centers
	
	print(init_w.shape)
	print(init_b.shape)
	print(init_centers.shape)

	return (init_w,init_b,init_centers)




def get_action_centers(center_file, hf, used_kmeans_list, sample_dim, k_clusters, alpha_state=False, input_norm=False, task='action', split=None):

	(init_w,init_b,init_centers) = get_centers(center_file, hf, used_kmeans_list, sample_dim, k_clusters, input_norm=input_norm,
		alpha_state=alpha_state, task=task, split=split)
	return (init_w,init_b,init_centers)





if __name__ == '__main__':
	train_split_file = '/home/lyb/XYJ/dataset/ucfTrainTestlist/trainlist01.txt'
	video_fea_root = '/mnt/lyb/xyj/UCF-101-5fps-pool5-feature'
	train_v_list = []
	with open(train_split_file,'r') as reader:
		for line in reader:
			temp = line.strip().split(' ')
			train_v_list.append(temp[0])

	train_v_list = np.random.permutation(train_v_list)
	used_train_list = train_v_list[0:10]


	n_samples = 256*len(used_train_list)
	sample_dim = 512
	k_clusters = 256
	n_iter = 10
	random_state = 43

	redu_dim = 256

	get_clsts_PCA_whiten(video_fea_root, used_train_list, sample_dim, k_clusters, redu_dim)