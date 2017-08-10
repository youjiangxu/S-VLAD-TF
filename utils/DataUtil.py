import numpy as np
import os
import re
import h5py
import math
import json




def get_data(split=1,file='/mnt/data3/xyj/data/hmdb/gt/hmdb51.json'):
	'''
	v2i = {'': 0, 'UNK':1}  # vocabulary to index
	'''
	split_tp = json.load(open(file,'r'))
	train_data = split_tp[split-1][0]
	test_data = split_tp[split-1][1]
	
	print('train %d, test %d' %(len(train_data),len(test_data)))
	return train_data, test_data


def getBatchVideoFeature(batch_data, hf, feature_shape):
	batch_size = len(batch_data)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')
	labels = np.zeros((batch_size,),dtype='int32')
	for idx, vid in enumerate(batch_data):
		feature = hf[vid[0]]
		input_video[idx] = np.reshape(feature,feature_shape)

		labels[idx]=vid[1]
	return input_video, labels

def getTestBatchVideoFeature(batch_data, hf, feature_shape):
	batch_size = len(batch_data)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')
	labels = np.zeros((batch_size,),dtype='int32')
	for idx, vid in enumerate(batch_data):
		feature = hf[vid[0]+'/0']
		input_video[idx] = np.reshape(feature,feature_shape)

		labels[idx]=vid[1]
	return input_video, labels

def getOversamplingBatchVideoFeature(batch_data, hf, feature_shape, modality='rgb'):
	batch_size = len(batch_data)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')
	labels = np.zeros((batch_size,),dtype='int32')
	for idx, vid in enumerate(batch_data):
		if modality=='rgb':
			temp = np.random.randint(0,10)
		else:
			temp = np.random.randint(0,10)
		feature = hf[vid[0]+'/'+str(temp)]
		input_video[idx] = np.reshape(feature,feature_shape)

		labels[idx]=vid[1]
	return input_video, labels



if __name__=='__main__':
	print('test')