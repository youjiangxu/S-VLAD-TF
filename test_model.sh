#!/bin/bash
pretrained_model=/mnt/data3/xyj/saved_model/hmdb51/bi_split2_seqvlad_google_k3_c64_redu512_d0.9_flow/lr0.0002_f10_B256/model/E33_L0.658308672053_A0.577124183007.ckpt
modality=flow
centers_num=64
split=2
test_output=/home/xyj/usr/local/saved_model/hmdb51_res/split_${split}_${modality}_${centers_num}.txt
python seqvlad_action_redu.py --step --bidirectional --epoch=20 --lr=0.0002 --centers_num=${centers_num} --reduction_dim=512 --dropout=0.9 --test --pretrained_model=${pretrained_model} --gpu_id=0 --modality=${modality} --test_output=${test_output} --split=${split}
