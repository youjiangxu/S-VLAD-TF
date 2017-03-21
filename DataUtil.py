import numpy as np
import string
import cPickle as cPickle





def get_video_conv_feature(hf, used_train_list, encoder_length=10,filter_num=512,filter_height=7,filter_width=7, frame_num=35):
    batch_size = len(used_train_list)
    input_vid_feature = np.zeros((batch_size,encoder_length,filter_num,filter_height,filter_width),dtype=np.float32)
    for i,sample in enumerate(used_train_list):
        vid_name = sample.strip().split('/')[1][:-4]
        start_pos = np.random.randint(0,frame_num-encoder_length)
        load_feature = hf[vid_name][start_pos:start_pos+10]

            # print(load_feature)
        input_vid_feature[i] = np.reshape(load_feature,(encoder_length,filter_num,filter_height,filter_width))
    # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
    return input_vid_feature

def load_batch_labels(vid_list,nb_classes,label_dict):
    batch_size = len(vid_list)
    video_labels = np.zeros((batch_size,nb_classes),dtype=np.int32)
    for idx,v_info in enumerate(vid_list):
        vid_name= v_info.strip().split('/')[1].split('_')[1]
        if vid_name=='HandStandPushups':
            vid_name='HandstandPushups'
        label = label_dict[vid_name]
        video_labels[idx][label-1] = 1
    return video_labels

def get_25_test_data(hf, used_train_list, encoder_length=10,filter_num=512,filter_height=7,filter_width=7, test_sub_volumn = 25):

    batch_size = len(used_train_list)
    input_vid_feature = np.zeros((test_sub_volumn,encoder_length,filter_num,filter_height,filter_width),dtype=np.float32)

    # for i,sample in enumerate(used_train_list):
    vid_name = used_train_list[0].strip().split('/')[1][:-4]

    for j in xrange(test_sub_volumn):
        input_vid_feature[j] = np.reshape(hf[vid_name][j:j+encoder_length],(encoder_length,filter_num,filter_height,filter_width))
    # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
    return input_vid_feature


'''5-crop rgb data'''

def get_video_conv_feature_5crop(hf, used_train_list, encoder_length=10,filter_num=512,filter_height=7,filter_width=7, frame_num=35):
    batch_size = len(used_train_list)
    input_vid_feature = np.zeros((batch_size,encoder_length,filter_num,filter_height,filter_width),dtype=np.float32)
    for i,sample in enumerate(used_train_list):
        vid_name = sample.strip().split('/')[1][:-4]
        
        start_crop = np.random.randint(0,5)
        start_pos = start_crop*frame_num+np.random.randint(0,frame_num-encoder_length)
        # print(start_crop)
        load_feature = hf[vid_name][start_pos:start_pos+10]

            # print(load_feature)
        input_vid_feature[i] = np.reshape(load_feature,(encoder_length,filter_num,filter_height,filter_width))
    # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
    return input_vid_feature

def get_25_test_data_5crop(hf, used_train_list, encoder_length=10,filter_num=512,filter_height=7,filter_width=7, test_sub_volumn=25,crop_num=5,frame_num=35):

    batch_size = len(used_train_list)
    input_vid_feature = np.zeros((test_sub_volumn*crop_num,encoder_length,filter_num,filter_height,filter_width),dtype=np.float32)

    # for i,sample in enumerate(used_train_list):
    vid_name = used_train_list[0].strip().split('/')[1][:-4]
    for crop_i in xrange(crop_num):
        for j in xrange(test_sub_volumn):
            idx = crop_i*test_sub_volumn+j
            feature_idx = crop_i*frame_num+j
            input_vid_feature[idx] = np.reshape(hf[vid_name][feature_idx:feature_idx+encoder_length],(encoder_length,filter_num,filter_height,filter_width))
    # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
    return input_vid_feature


'''10-crop rgb data'''

def get_video_conv_feature_10crop(hf1, hf2, used_train_list, encoder_length=10,filter_num=512,filter_height=7,filter_width=7, frame_num=35):
    batch_size = len(used_train_list)
    input_vid_feature = np.zeros((batch_size,encoder_length,filter_num,filter_height,filter_width),dtype=np.float32)

    for i,sample in enumerate(used_train_list):
        vid_name = sample.strip().split('/')[1][:-4]

        flip_state = np.random.randint(0,2)
        if flip_state==0:
            start_crop = np.random.randint(0,5)
            start_pos = start_crop*frame_num+np.random.randint(0,frame_num-encoder_length)
            load_feature = hf1[vid_name][start_pos:start_pos+10]
        else:
            start_crop = np.random.randint(0,5)
            start_pos = start_crop*frame_num+np.random.randint(0,frame_num-encoder_length)
            load_feature = hf2[vid_name][start_pos:start_pos+10]

        input_vid_feature[i] = np.reshape(load_feature,(encoder_length,filter_num,filter_height,filter_width))
    return input_vid_feature




# def get_caption(batch_caption,vocab,decoder_length=17):
#     batch_size = len(batch_caption)
#     input_caption = np.zeros((batch_size,decoder_length),dtype=np.int32)
#     vocab_size = len(vocab)
#     bos = vocab_size+1
#     eos = vocab_size+2
    
#     input_caption[:,0]=bos
#     # print(batch_caption)
#     for i,captions_info in enumerate(batch_caption):
#         for vid,captions in captions_info.items():
#             # print captions
#             for j,word in enumerate(captions):
#                 if j+1 < decoder_length:
#                     if vocab.has_key(word):
#                         input_caption[i][j+1]=vocab[word]
#                     else:
#                         input_caption[i][j+1]=vocab['UNK']
#             # print(input_caption[i])
#     # print input_caption

#     labels = np.zeros((batch_size,decoder_length,vocab_size+2),dtype='int32')


#     for i,sentence in enumerate(input_caption):
#         for j,word in enumerate(sentence):
#             if j>=1 :
#                 if word != 0:
#                     labels[i,j-1,word-1] = 1
#                 elif word == 0 :
#                     labels[i,j-1,eos-1] = 1
#                     break
#     # print(np.sum(np.sum(labels,axis=2),axis=1))
#     return input_caption,labels

# # def get_test_caption(batch_caption,vocab,decoder_length=17):
# #     batch_size = len(batch_caption)
# #     input_caption = np.zeros((batch_size,decoder_length),dtype=np.int32)
# #     vocab_size = len(vocab)
# #     bos = vocab_size+1
# #     eos = vocab_size+2
    
# #     input_caption[:,0]=bos
# #     # print(np.sum(np.sum(input_caption,axis=2),axis=1))
    
# #     return input_caption

# def get_video_fc_feature(batch_caption,hf, encoder_length=10,filter_num=1024):
#     batch_size = len(batch_caption)
#     input_vid_feature = np.zeros((batch_size,encoder_length,filter_num),dtype=np.float32)
#     for i,captions in enumerate(batch_caption):
#         for vid,text in captions.items():
#             load_feature = hf[vid]
#             # print(load_feature)
#             input_vid_feature[i] = np.reshape(load_feature,(encoder_length,filter_num))
#     # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
#     return input_vid_feature
    

# def get_video_conv_feature(batch_caption,hf, encoder_length=10,filter_num=1024,filter_height=7,filter_width=7):
#     batch_size = len(batch_caption)
#     input_vid_feature = np.zeros((batch_size,encoder_length,filter_num,filter_height,filter_width),dtype=np.float32)
#     for i,captions in enumerate(batch_caption):
#         for vid,text in captions.items():
#             load_feature = hf[vid]
#             # print(load_feature)
#             input_vid_feature[i] = np.reshape(load_feature,(encoder_length,filter_num,filter_height,filter_width))
#     # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
#     return input_vid_feature

# def get_test_video_conv_feature(batch_caption,hf, encoder_length=10,filter_num=1024,filter_height=7,filter_width=7):
#     batch_size = len(batch_caption)
#     input_vid_feature = np.zeros((batch_size,encoder_length,filter_num,filter_height,filter_width),dtype=np.float32)
#     # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
#     for i,vid in enumerate(batch_caption):
        
#         load_feature = hf[vid]
#         # print(np.sum(load_feature))
        
        
#         input_vid_feature[i] = np.reshape(load_feature,(encoder_length,filter_num,filter_height,filter_width))
#         # print(np.sum(input_vid_feature[i]))
#         # break
#     # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
#     return input_vid_feature


# def get_test_video_feature(batch_caption,hf, encoder_length=10,filter_num=1024):
#     batch_size = len(batch_caption)
#     input_vid_feature = np.zeros((batch_size,encoder_length,filter_num),dtype=np.float32)
#     # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
#     for i,vid in enumerate(batch_caption):
        
#         load_feature = hf[vid]
#         # print(np.sum(load_feature))
        
        
#         input_vid_feature[i] = np.reshape(load_feature,(encoder_length,filter_num))
#         # print(np.sum(input_vid_feature[i]))
#         # break
#     # print(np.sum(input_vid_feature[0]-input_vid_feature[1]))
#     return input_vid_feature

    
if __name__ == "__main__":
    print('hello')