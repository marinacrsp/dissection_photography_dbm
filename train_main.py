"""
Train a diffusion model on images.
"""

import argparse
import torch

from datasets import load_data
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir,
)
from ddbm import dist_util, logger
from pathlib import Path
import wandb
from glob import glob
import os

def main(args):
    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)

    device='cuda:0'
    logger.configure(dir=workdir)

    name = args.exp if args.resume_checkpoint == "" else args.exp + "_resume"
    wandb.init(
        project="dbi_definite",
        group=args.exp,
        name=name,
        config=vars(args),
        mode="online",
    )
    logger.log("creating model and diffusion...")

    data_image_size = args.image_size

    # Load target model
    resume_train_flag = False

    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f"{workdir}/*model*[0-9].*"))

        if args.pretrained_ckpt is not None:
            max_ckpt = args.pretrained_ckpt
            args.resume_checkpoint = max_ckpt
    
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    traindata, valdata = load_data(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        num_volumes=args.num_volumes,
        val_bsz=args.val_bsz,
        dataset=args.dataset,
        spacing_limits=args.space_limits,
        n_slices_per_sandwich=args.n_slices_per_sandwich,
        local_batch_size=args.batchsize,
        image_size=data_image_size,
        num_workers=args.num_workers,
        loss=args.loss_type,
    )

    from ddbm.validate_epoch import ValidationLoop

    if "contin" in args.loss_type:
        from diffusion_bridges_dissection_photography.ddbm.train_util_continuity import TrainLoop
    elif "percept" in args.loss_type:
        # from ddbm.train_util_perceptual_singlegpu_wresidual import TrainLoop
        from ddbm.train_util_perceptual_singlegpu import TrainLoop
    else:
        from ddbm.train_util_singlegpu import TrainLoop
        
    # run until it gets stopped
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=traindata,
        val_data=valdata,
        batch_size=args.batchsize,
        lr=args.lr,
        ema_rate=args.ema_rate,
        loss_type=args.loss_type,
        c_weight=args.c_weight,
        p_weight=args.p_weight,
        log_interval=args.log_interval,
        validate_every=args.validate_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        train_mode=args.train_mode,
        resume_train_flag=resume_train_flag,
        n_slices_persand = args.n_slices_per_sandwich,
        **sample_defaults(),
    ).run_loop()



def create_argparser():
    defaults = dict(
        unet_type="adm",
        interpolate=True,\
        num_volumes=1,
        data_dir="",
        val_dir="",
        dataset="edges2handbags",
        schedule_sampler="real-uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        in_channels=4,
        batchsize=-1,
        val_bsz=20,
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=500,
        validate_interval=1000,
        test_interval=500,
        save_interval=5000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp="",
        use_fp16=True,
        use_dist_conditioning=False,
        fp16_scale_growth=1e-3,
        num_workers=0,
        use_augment=False,
        pretrained_ckpt=None,
        train_mode="ddbm",
        loss_type="L2",
        c_weight=0.0,
        p_weight=0.0,
        space_limits="6,12",
        n_slices_per_sandwich=1,
        
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
