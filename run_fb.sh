#!/bin/bash
#python seqvlad_action_redu.py --step --bidirectional --epoch=20 --lr=0.0005 --centers_num=32 --reduction_dim=512 --dropout=0.9
#python seqvlad_action_redu.py --bidirectional --epoch=40 --lr=0.0002 --centers_num=16 --reduction_dim=512 --dropout=0.9 --modality=flow

#python seqvlad_fb_action.py --bidirectional --epoch=60 --lr=0.0001 --centers_num=64 --reduction_dim=512 --dropout=0.5 --modality=rgb --split=1 --gpu_id=1 --model=seqvlad --redu_filter_size=1 --feature=google --model=seqvlad_fb_v2
#python seqvlad_fb_sep_action.py --bidirectional --epoch=60 --lr=0.0001 --centers_num=64 --reduction_dim=512 --dropout=0.2 --modality=rgb --split=1 --gpu_id=1 --model=seqvlad --redu_filter_size=1 --feature=google --model=seqvlad_fb_v2 \
# | tee /home/xyj/usr/local/log/sep_featnorm_.0001reg.txt 

python seqvlad_fb_tfrecords.py --bidirectional --epoch=100 --lr=0.0001 --centers_num=64 --reduction_dim=512 --dropout=0.2 --modality=rgb --split=2 --gpu_id=1 --redu_filter_size=1 --feature=google --model=seqvlad_fb_v3
##python seqvlad_fb_tfrecords.py --bidirectional --epoch=60 --lr=0.0001 --centers_num=64 --reduction_dim=512 --dropout=0.2 --modality=rgb --split=3 --gpu_id=1 --redu_filter_size=1 --feature=google --model=seqvlad_fb_v3
#python seqvlad_fb_tfrecords.py --bidirectional --epoch=100 --lr=0.0001 --centers_num=64 --reduction_dim=512 --dropout=0.2 --modality=flow --split=2 --gpu_id=1 --redu_filter_size=1 --feature=google --model=seqvlad_fb_v3

#python seqvlad_fb_action.py --bidirectional --epoch=60 --lr=0.0001 --centers_num=64 --reduction_dim=512 --dropout=0.2 --modality=rgb --split=1 --gpu_id=1 --model=seqvlad --redu_filter_size=1 --feature=google --model=seqvlad_fb_v1
#python seqvlad_fb_action.py --bidirectional --epoch=60 --lr=0.0001 --centers_num=128 --reduction_dim=512 --dropout=0.1 --modality=rgb --split=1 --gpu_id=1 --model=seqvlad --redu_filter_size=1 --feature=google --model=seqvlad_fb_v1 | tee /home/xyj/usr/local/log/d0.1.txt



