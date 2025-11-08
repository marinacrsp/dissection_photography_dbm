#!/bin/bash
export PATH=/usr/bin/python3.11:$PATH
export PYTHONPATH=/usr/bin/python3.11:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./


DATASET_NAME=brain
TRAIN_MODE=dbi #dist_interp_concat_resume #ddbm_dist_interp_concat_loss

source /homes/1/ma1282/mar/CADD_gitrepo/ddpm_m/bin/activate # the repository

FREQ_SAVE_ITER=5000


## ARCHITECTURE
CHNL_MULT="1,2,2" #"1,2,2"
NUM_CH=128 
NUM_RES_BLOCKS=2
ATTN_TYPE=True 
IN_CHANNELS=1
SAVE_ITER=5000 #every 5000 steps
DROPOUT=0.1
DIST_COND=True

## DATASET
DATASET=brain
DATA_DIR=/autofs/vast/lemon/data_curated_hemis/brain_mris_QCed/
VAL_DIR=/homes/1/ma1282/marina_almeria/dbi/workdir/validation_normalized_stack.pth
IMG_SIZE=160
BATCHSIZE=8 #4 #25 #this corresponds to the number of sandwiches sampled per volume
NVOLS=5 #12
VAL_BATCHSIZE=32
SPACE_LIMS="2,12"
SLICES_PER_SAND=1

## OPTIMIZATION
NWORKERS=2
PRED="vp" 
BETA_MAX=1.0
BETA_MIN=0.1
SIGMA_MAX=1
SIGMA_MIN=0.0001
LOSS="L1_gradient"
# LOSS="L1"
C_WEIGHT=1
P_WEIGHT=1


UNET="vanilla"
INTERP=True
EXP=${TRAIN_MODE}-${LOSS}-${UNET}-Interp-${INTERP}-${PRED}-slabs${BATCHSIZE}_x_vols${NVOLS}_small
CKPT=/homes/1/ma1282/marina_almeria/dbi/workdir/dbi-L1_gradient_perceptual-vanilla-Interp-True-vp-slabs10_x_vols4_small/ckpt_s10000.pt
python $run_args train_single_gpu.py --exp=$EXP \
    --unet_type $UNET \
    --interpolate $INTERP \
    --num_workers $NWORKERS \
    --num_volumes $NVOLS \
    --dropout $DROPOUT --batchsize $BATCHSIZE --space_limits=$SPACE_LIMS --n_slices_per_sandwich=$SLICES_PER_SAND \
    --val_bsz $VAL_BATCHSIZE \
    --image_size $IMG_SIZE  --num_channels $NUM_CH  --in_channels $IN_CHANNELS --channel_mult=$CHNL_MULT \
    --num_res_blocks $NUM_RES_BLOCKS --use_dist_conditioning=$DIST_COND \
    --noise_schedule=$PRED \
    --loss_type=$LOSS --c_weight=$C_WEIGHT --p_weight=$P_WEIGHT \
    --use_new_attention_order $ATTN_TYPE \
    ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"}  \
    --data_dir=$DATA_DIR --dataset=$DATASET  --val_dir=$VAL_DIR \
    --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
    --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER \
    --pretrained_ckpt="${CKPT}" \
    ${CKPT:+ --resume_checkpoint="${CKPT}"} \


# done