#!/bin/bash
export PATH=/usr/bin/python3.11:$PATH
export PYTHONPATH=/usr/bin/python3.11:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./


DATASET_NAME=brain
TRAIN_MODE=dbi #dist_interp_concat_resume #ddbm_dist_interp_concat_loss

source /homes/1/ma1282/mar/CADD_gitrepo/ddpm_m/bin/activate # the repository

FREQ_SAVE_ITER=5000


## ARCHITECTURE
CHNL_MULT="1,2,3,5" #"1,2,2"
NUM_CH=192 
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
BATCHSIZE=6 #this corresponds to the number of sandwiches sampled per volume
VAL_BATCHSIZE=32
SPACE_LIMS="2,12"
SLICES_PER_SAND=1

## OPTIMIZATION
NWORKERS=0
BETA_MAX=1.0
BETA_MIN=0.1
SIGMA_MAX=1
SIGMA_MIN=0.0001
EXP=${TRAIN_MODE}-${LOSS}-${UNET}-${PRED}
UNET="vanilla"
for ckpt_id in {130..290..5}; do
    CKPT=/homes/1/ma1282/marina_almeria/dbi/workdir/dbi-L1_gradient_perceptual-vanilla-Interp-True-vp-slabs4_x_vols6_residual/ckpt_s${ckpt_id}000.pt
    echo ${CKPT}
    python $run_args validate.py --exp=$EXP \
        --unet_type $UNET \
        --num_workers $NWORKERS \
        --dropout $DROPOUT --space_limits=$SPACE_LIMS --n_slices_per_sandwich=$SLICES_PER_SAND \
        --val_bsz $VAL_BATCHSIZE \
        --image_size $IMG_SIZE  --num_channels $NUM_CH  --in_channels $IN_CHANNELS --channel_mult=$CHNL_MULT \
        --num_res_blocks $NUM_RES_BLOCKS --use_dist_conditioning=$DIST_COND \
        --noise_schedule=$PRED \
        --use_new_attention_order $ATTN_TYPE \
        ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"}  \
        --data_dir=$DATA_DIR --dataset=$DATASET  --val_dir=$VAL_DIR \
        --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
        --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER \
        --pretrained_ckpt="${CKPT}" \
        ${CKPT:+ --resume_checkpoint="${CKPT}"} 
done

