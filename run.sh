#!/bin/bash
#python seqvlad_action_redu.py --step --bidirectional --epoch=20 --lr=0.0005 --centers_num=32 --reduction_dim=512 --dropout=0.9
#python seqvlad_action_redu.py --bidirectional --epoch=40 --lr=0.0002 --centers_num=16 --reduction_dim=512 --dropout=0.9 --modality=flow

#python seqvlad_action_redu.py --bidirectional --epoch=100 --lr=0.0002 --centers_num=64 --reduction_dim=512 --dropout=0.1 --modality=rgb --split=1 --gpu_id=1
python seqvlad_action_redu.py --bidirectional --epoch=100 --lr=0.0002 --centers_num=64 --reduction_dim=512 --dropout=0.1 --modality=flow --split=1 --gpu_id=1
python seqvlad_action_redu.py --bidirectional --epoch=100 --lr=0.0002 --centers_num=64 --reduction_dim=512 --dropout=0.1 --modality=flow --split=2 --gpu_id=1
python seqvlad_action_redu.py --bidirectional --epoch=100 --lr=0.0002 --centers_num=64 --reduction_dim=512 --dropout=0.1 --modality=flow --split=3 --gpu_id=1
#python seqvlad_action_redu.py --bidirectional --epoch=40 --lr=0.0002 --centers_num=64 --reduction_dim=512 --dropout=0.9 --modality=flow --split=2
