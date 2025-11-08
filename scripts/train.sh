export PATH=/usr/bin/python3.11:$PATH
export PYTHONPATH=/usr/bin/python3.11:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./
# export LD_LIBRARY_PATH=/usr/lib64/python3.11:$LD_LIBRARY_PATH
# export PATH=/usr/pubsw/packages/CUDA/12.2/bin:$PATH
# export CPLUS_INCLUDE_PATH=/usr/pubsw/packages/CUDA/12.2/include:$CPLUS_INCLUDE_PATH

DATASET_NAME=brain
TRAIN_MODE=dbi #dist_interp_concat_resume #ddbm_dist_interp_concat_loss

source /homes/1/ma1282/mar/CADD_gitrepo/ddpm_m/bin/activate # the repository

FREQ_SAVE_ITER=5000

export CUDA_VISIBLE_DEVICES="0,1" #,2
run_args="--nproc_per_node 2 \
           --master_port 20600"

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
IMG_SIZE=160
BATCHSIZE=28 #this corresponds to the number of sandwiches sampled per volume
SPACE_LIMS="2,12"
SLICES_PER_SAND=1


## OPTIMIZATION
NWORKERS=6
PRED="vp"
BETA_D=2
BETA_MIN=0.1
SIGMA_MAX=1
LOSS="L1_gradient_perceptual"
C_WEIGHT=1
P_WEIGHT=1
SIGMA_MIN=0.0001

UNET="vanilla"
EXP=${TRAIN_MODE}-${LOSS}-${UNET}-1

CKPT=/homes/1/ma1282/marina_almeria/dbi/workdir/dbi-L1_gradient_perceptual-dist_cond-Interp-True-vp-slabs5_x_vols5/ckpt_s5000.pt

torchrun $run_args train.py --exp=$EXP \
 --unet_type $UNET \
 --num_workers $NWORKERS \
 --dropout $DROPOUT --batchsize $BATCHSIZE --space_limits=$SPACE_LIMS --n_slices_per_sandwich=$SLICES_PER_SAND \
 --image_size $IMG_SIZE  --num_channels $NUM_CH  --in_channels $IN_CHANNELS --channel_mult=$CHNL_MULT \
 --num_res_blocks $NUM_RES_BLOCKS --use_dist_conditioning=$DIST_COND \
 --noise_schedule=$PRED \
 --loss_type=$LOSS --c_weight=$C_WEIGHT --p_weight=$P_WEIGHT \
 --use_new_attention_order $ATTN_TYPE \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"}  \
 --data_dir=$DATA_DIR --dataset=$DATASET  \
 --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
 --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER \
 --pretrained_ckpt="${CKPT}" \
 ${CKPT:+ --resume_checkpoint="${CKPT}"} \