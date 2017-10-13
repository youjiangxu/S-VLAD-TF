#!/bin/bash
pretrained_model=/mnt/data3/xyj/saved_model/seqvlad_fb_v3_hmdb51/bi_hmdb51_split2_seqvlad_fb_v3_google_k3_c64_redu512_d0.2_rgb_rfs1/lr0.0001_f10_B128/model/E60_TaL0.654467740229.ckpt
modality=rgb
centers_num=64
split=2
model=seqvlad
redu_filter_size=1
feature=google
dataset=hmdb51
model=seqvlad_fb_v3
test_output=/home/xyj/usr/local/saved_model/${dataset}_res/${feature}_${model}_split_${split}_${modality}_${centers_num}_rfs${redu_filter_size}_m${model}_att50_E60.txt
python seqvlad_fb_tfrecords.py  --bidirectional --epoch=20 --lr=0.0002 --centers_num=${centers_num} --reduction_dim=512 --dropout=0.9 --test --pretrained_model=${pretrained_model} --gpu_id=1 --modality=${modality} --test_output=${test_output} --split=${split} \
 --model=${model} --redu_filter_size=${redu_filter_size} --feature=${feature} --reduction_dim=512 --dataset=${dataset} --model=${model}
