from six.moves import cPickle
import sys, json, time, os
import numpy as np
import DataUtil

import h5py


print('preparing parameter ...')


def test_merged_model(temporal_pkl, spatial_pkl, test_v_list, temp_coff, spat_coff, label_dict):
	ave_acc = 0
	total_test_v_num = len(test_v_list)
	nb_classes = 101
	for idx,vid in enumerate(test_v_list):
		f = open(temporal_pkl+'/'+vid[0:-4]+'.pkl', 'rb')
		temporal_predict = cPickle.load(f)
		f.close()

		f = open(spatial_pkl+'/'+vid[0:-4]+'.pkl', 'rb')
		spatial_predict = cPickle.load(f)
		f.close()

		input_label = DataUtil.load_batch_labels([vid],nb_classes,label_dict)

		final_predict = temp_coff*temporal_predict+spat_coff*spatial_predict

		predict_video = np.argmax(final_predict)
		ground_truth = np.argmax(input_label)

		if(predict_video==ground_truth):
			ave_acc+=1
		# print('predict class:%d, ground class:%d, right video num:%d/%d' %(predict_video, ground_truth, ave_acc, (idx+1)))
	ave_acc = ave_acc*1.0 /total_test_v_num
	print('test merged model with temp_coff vs. spat_coff =(%.5f  vs %.5f), acc%.5f' %(temp_coff,spat_coff,ave_acc))

	
if __name__ == '__main__':
	split = 2 
	test_split_file = '/home/xyj/usr/local/data/ucf-101/ucfTrainTestlist/testlist0'+str(split)+'.txt'

	temporal_pkl = '/home/xyj/usr/local/predict_result/flow_split2/K16_d0.9_lr0.5'
	spatial_pkl = '/home/xyj/usr/local/predict_result/rgb_split2/K16_d0.9_lr0.05iter113_L1.1204_A0.7516'

	label_file='/home/xyj/usr/local/data/ucf-101/ucfTrainTestlist/classInd.txt'


	test_v_list = []
	
	with open(test_split_file,'r') as reader:
		for line in reader:
			temp = line.strip().split(' ')
			test_v_list.append(temp[0])

	label_dict = {}
	with open(label_file,'r') as reader:
		for line in reader:
			temp = line.strip().split(' ')
			label_dict[temp[1]]=int(temp[0])

	ratio = [0.01*i for i in xrange(101)]

	for temp_coff in ratio:
	
		spat_coff =  1 - temp_coff
		test_merged_model(temporal_pkl, spatial_pkl, test_v_list, temp_coff, spat_coff, label_dict)


