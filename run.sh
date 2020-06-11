#!/usr/bin/env bash
train_batch_size=64
devices=[0,1,2,3]
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 100
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 200
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 300
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 500
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 3e-5 --num_warmup_steps 200
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 5e-5 --num_warmup_steps 200
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 1e-5 --num_warmup_steps 200
CUDA_VISIBLE_DEVICES=$devices python train.py --train_batch_size $train_batch_size --shuffle --lr 1e-4 --num_warmup_steps 200