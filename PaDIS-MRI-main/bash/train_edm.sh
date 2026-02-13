# !/bin/bash
# Train EDM on FastMRI brain dataset

CUDA_VISIBLE_DEVICES=0
NPROC=1
ANATOMY=brain                               # brain
DATA=noisy                                  # FastMRI data inherently has some noise
SNR=32dB                                    # 32dB SNR (average PSNR in FastMRI brain dataset) – indicates no additional noise added
ROOT_OUTDIR=/home/rohan/EDM-FastMRI         # root directory of where to save model checkpoints
ROOT_DATA=/data/rohan/brain_train_d384_s200 # path to the train dataset
BATCH_SIZE=4                                # use 4 or 8 depending on GPU memory
PRECOND=edm

torchrun --standalone --nproc_per_node=$NPROC train/fastmri-edm/train.py \
 --outdir=$ROOT_OUTDIR/$ANATOMY/$SNR \
 --data=$ROOT_DATA/$SNR \
 --cond=0 --arch=ddpmpp --duration=10 \
 --batch=$BATCH_SIZE --cbase=128 --cres=1,1,2,2,2,2,2 \
 --lr=5e-5 --ema=0.5 --dropout=0.05 \
 --desc=container_test --tick=1 --snap=50 \
 --dump=200 --seed=2025 --precond=$PRECOND --augment=0 \
 --normalize=0 --loader=Numpy --gpu=$CUDA_VISIBLE_DEVICES 