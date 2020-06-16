#!/usr/bin/env bash
train_batch_size=8
devices=0,1,2,3,6,7,8,9
#CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 100 --fold 0 --fp16
#CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 100 --fold 1 --fp16
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 100 --fold 2 --fp16
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 100 --fold 3 --fp16
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 100 --fold 4 --fp16
