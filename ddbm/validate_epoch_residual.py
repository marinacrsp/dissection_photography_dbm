import copy
import functools
import os
import pickle
import numpy as np
import gc
import blobfile as bf
import torch
from torch.optim import RAdam
import pandas as pd
import time
from . import logger
from .nn import update_ema, requires_grad
from torch.nn import L1Loss
from ddbm.karras_diffusion import karras_sample
from ddbm.random_util import get_generator
import wandb
import glob
import matplotlib.pyplot as plt

class ValidationLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        val_data,
        batch_size,
        lr,
        ema_rate,
        loss_type,
        log_interval, #how often do you see the outputs
        test_interval, #how often do you compute the test 
        save_interval, # how often do you save the ckpt !!! IMPORTANT $SAVE_ITER
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        c_weight=0.0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=5001, #TODO
        augment_pipe=None,
        train_mode="ddbm",
        resume_train_flag=False,
        n_slices_persand=1,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.valdata = val_data
        self.batch_size = batch_size
        self.lr = lr
        self.workdir = workdir
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = False
        self.fp16_scale_growth = False
        self.ddp_model = self.model
        self.device='cuda:0'
        self.train_mode = train_mode
        self.step = 0
        self.resume_train_flag = resume_train_flag
        self.last_step = self.step
        
        self._load_and_sync_parameters()
        self.generator = get_generator(sample_kwargs["generator"], self.batch_size, 42)
        self.sample_kwargs = sample_kwargs
        self.n_slices_persand = n_slices_persand
        self.dist_scale = 0.1
        self.siz = [160, 160]
        self.augment = augment_pipe

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            ckpt = torch.load(resume_checkpoint, weights_only=True)["model"]
            self.last_step = torch.load(resume_checkpoint, weights_only=True)["step"]
            print(f"Training from step: {self.last_step}")
            self.ddp_model.load_state_dict(ckpt)
            self.ddp_model.to(self.device)

    def _load_optimizer_state(self):
        main_checkpoint= self.resume_checkpoint[:self.resume_checkpoint.find("ckpt")] 
        print(main_checkpoint)
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt.pt")
        print(opt_checkpoint)
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            try:
                state_dict = torch.load(opt_checkpoint, map_location=self.device)
                self.opt.load_state_dict(state_dict)
            except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
                logger.warn(f"Failed to load optimizer state from {opt_checkpoint}: {e}")
                logger.warn("Proceeding with a fresh optimizer state.")
        else:
            logger.log(f"No optimizer checkpoint found at {opt_checkpoint}. Starting fresh.")

    def run_loop(self):

        
        model_name = os.path.basename(os.path.dirname(self.resume_checkpoint))
        csv_path = os.path.join(self.resume_checkpoint[:self.resume_checkpoint.find(model_name)], "validation_scores.csv")
        ckpt_str = os.path.basename(self.resume_checkpoint)
        ckpt_iter = int(ckpt_str.split("ckpt_s")[1].split(".pt")[0])
        ckpt_col = f"{ckpt_iter//1000}k"

        # Load existing table or create new
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
        else:
            df = pd.DataFrame()
        self.val_loss = self.validate()

        # Ensure model row exists
        if ckpt_col not in df.columns:
            df[ckpt_col] = None
        if model_name not in df.index:
            df.loc[model_name] = [None] * len(df.columns)

        df.at[model_name, ckpt_col] = self.val_loss

        df = df.reindex(sorted(df.columns, key=self.sort_key), axis=1)
        
        df.to_csv(csv_path)
        print(f"Validation score {self.val_loss:.6f} saved to {csv_path}")
        
    def sort_key(self, c):
        try:
            return int(c.replace("k", ""))
        except:
            return float('inf')

    def validate(self):         
            self.ddp_model.convert_to_fp32()
            self.ddp_model.eval()
            
            val_loss_step = 0.0
            for idx, val_batch in enumerate(self.valdata):
                inputs, dists_th, outputs = val_batch[:,:2], val_batch[:,2:4], val_batch[:,-1]
                outputs = outputs.unsqueeze(1)
                w1 = dists_th[:, 1] / (dists_th.sum(dim=1))
                w2 = 1 - w1

                xT = (w1 * inputs[:, 0] + w2 * inputs[:, 1]).unsqueeze(1) # shape bsz, 160, 160
                slab = torch.concat([inputs, self.dist_scale * dists_th], dim=1)
                dists = dists_th[:,:,0,0]
                cond = {"xT": xT, "slab" : slab, "dists": dists*self.dist_scale, "gradient_gt": None}
                xT = xT.to(self.device, dtype=torch.float32)
                batch = outputs.to(self.device, dtype=torch.float32) #output
                cond = {
                    k: (v.to(self.device, dtype=torch.float32) if v is not None else None)
                    for k, v in cond.items()
                }
                with torch.autocast(device_type="cuda", enabled=False):
                    residual, path, nfe, pred_x0, sigmas, _ = karras_sample(
                                                                                self.diffusion,
                                                                                self.ddp_model,
                                                                                xT, #prior distribution
                                                                                None, #x0,
                                                                                steps=10,
                                                                                mask=None,
                                                                                model_kwargs=cond,
                                                                                device=self.device,
                                                                                clip_denoised=True,
                                                                                sampler="heun",
                                                                                churn_step_ratio=0.0,
                                                                                eta=0.0,
                                                                                order=1,
                                                                                seed=42,
                                                                            )
                    prediction = residual + cond["xT"]
                    val_loss = L1Loss()(prediction,batch)
                    val_loss_step += val_loss.item()
        
            val_loss_step /= len(self.valdata)
            print(f'Validation loss cohort: {val_loss_step}')            
            return val_loss_step


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None
    