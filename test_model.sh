#!/bin/bash
pretrained_model=/mnt/data3/xyj/saved_model/hmdb51/bi_new_split1_seqvlad_google_k3_c16_redu512_d0.5_rgb/lr0.0001_f10_B128/model/E39_TaL1.0475540672_TeL2.09324229757_A0.525490196078.ckpt
modality=rgb
centers_num=16
split=1
model=seqvlad
test_output=/home/xyj/usr/local/saved_model/hmdb51_res/${model}_split_${split}_${modality}_${centers_num}.txt
python seqvlad_action_redu.py --step --bidirectional --epoch=20 --lr=0.0002 --centers_num=${centers_num} --reduction_dim=512 --dropout=0.9 --test --pretrained_model=${pretrained_model} --gpu_id=1 --modality=${modality} --test_output=${test_output} --split=${split} \
 --model=${model}
