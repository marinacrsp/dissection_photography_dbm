#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./

# Batch size per GPU
BS=1

# Dataset and checkpoint
DATASET_NAME=brain
SPLIT=test
SAVE_PATH=/homes/1/ma1282/mar/DiffusionBridge/workdir/samples

source scripts/args.sh $DATASET_NAME
source /homes/1/ma1282/mar/CADD_gitrepo/ddpm_m/bin/activate

# Number of function evaluations (NFE)
NFE=$2
echo $IMG_SIZE

## ARCHITECTURE
CHNL_MULT="1,2,2" #"1,2,3,5"
NUM_CH=128 #192
NUM_RES_BLOCKS=2
ATTN_TYPE=True 
IN_CHANNELS=1
DROPOUT=0.1

# Sampler
BETA_D=2
BETA_MIN=0.1
SIGMA_MAX=1
SIGMA_MIN=0.0001

N=10
SAMPLER="heun"
ETA=0.0
echo $N
MODEL_PATH=/homes/1/ma1282/marina_almeria/dbi/workdir/dbi-L1_gradient_perceptual-vanilla-Interp-True-vp-slabs8_x_vols5_small/ckpt_s90000.pt
DIST_COND=True
UNET=vanilla

# for d in '4' '8' '12'; do
d=8
INPUT_FILE=/homes/1/ma1282/marina_almeria/photo_recon_uw/00_photo_recon/17-0333/photo_recon_${d}mm_left.nii.gz

python $run_args inference_photo_real_batch.py \
    --steps $N \
    --unet_type $UNET --input_file $INPUT_FILE \
    --image_size $IMG_SIZE  --num_channels $NUM_CH  --in_channels $IN_CHANNELS --channel_mult=$CHNL_MULT \
    --model_path $MODEL_PATH --batchsize $BATCHSIZE --sampler $SAMPLER \
    ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \
    --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --save_path $SAVE_PATH \
    --use_dist_conditioning=$DIST_COND \
    --dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH --in_channels $IN_CHANNELS --channel_mult=$CHNL_MULT --num_res_blocks $NUM_RES_BLOCKS \
    --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT \
    ${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} \
    ${ETA:+ --eta="${ETA}"} \
    ${ORDER:+ --order="${ORDER}"} 

# done