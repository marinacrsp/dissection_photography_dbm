import copy
import functools
import os
import pickle
import numpy as np
import gc
import blobfile as bf
import torch
from torch.optim import RAdam
import time
from . import logger
from .nn import update_ema, requires_grad
from torch.nn import L1Loss
from ddbm.karras_diffusion import karras_sample
from ddbm.random_util import get_generator
import wandb
import glob
import matplotlib.pyplot as plt

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
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
        self.data = train_data
        self.valdata = val_data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps
        self.loss_type = loss_type
        self.continuity_weight = c_weight


        self.ddp_model = self.model
        self.device='cuda:0'
        

        self.opt = RAdam(self.ddp_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train_mode = train_mode
        self.step = 0
        self.resume_train_flag = resume_train_flag
        self.last_step = self.step
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        self.cumul_loss = 0.0
        self.contin_loss = 0.0
        
        self._load_and_sync_parameters()
        self._load_optimizer_state()
        self.ema = copy.deepcopy(self.model).to(self.device)
        requires_grad(self.ema, False)

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
        logger.log(f'Training with {self.loss_type} loss')
        while True:
            self.val_loss = self.validate()
            self.ddp_model.train()
            
            break
            self.ddp_model.convert_to_fp16()
            for batch in self.data:

                inputs, gradient_imgs, dists, outputs = [obj.squeeze(0) for obj in batch]
                inputs = inputs.unsqueeze(1).repeat(1, self.n_slices_persand, 1, 1, 1)
                inputs = inputs.reshape(-1, *inputs.shape[2:]) # b*slices/slab, 2, H, W 

                outputs = outputs.reshape(-1, *outputs.shape[2:]).unsqueeze(1) # b*slices/slab, 1, H, W 
                gradient_imgs = gradient_imgs.reshape(-1, *outputs.shape[2:]).unsqueeze(1)
                dist_flat = dists.reshape(-1,*dists.shape[2:])

                dists_th = dist_flat[..., None, None].repeat(1, 1, *self.siz) 
                w1 = dists_th[:, 1] / (dists_th.sum(dim=1))
                w2 = 1 - w1

                xT = (w1 * inputs[:, 0] + w2 * inputs[:, 1]).unsqueeze(1) # shape bsz, 160, 160
                slab = torch.concat([inputs, self.dist_scale * dists_th], dim=1)

                if self.step >= self.total_training_steps: 
                    self.save()
                    print('MAX # TRAINING STEPS REACHED, TERMINATING LOOP')
                    return False

                
                cond = {"xT": xT, "slab" : slab, "dists": dist_flat*self.dist_scale, "gradient_gt": gradient_imgs}
                took_step = self.run_step(outputs, cond)

                if took_step and self.step % self.log_interval == 0:
                    wandb.log({'train/loss': self.loss_acc}, step=self.step+self.last_step)
                    wandb.log({'train/loss_continuity': self.loss_cont_acc}, step=self.step+self.last_step)

                if took_step and self.step % self.save_interval == 0: #if took_step is True and we're in save_interval mode
                    self.save() 
                    torch.cuda.empty_cache()

    def run_step(self, batch, cond): #training step of the validation data
        self.forward_backward(batch, cond) # inputs: x0, xT
        # logger.logkv_mean("lg_loss_scale", np.log2(self.scaler.get_scale()))

        self.step += 1
        update_ema(self.ema, self.model)
        self._anneal_lr()

        return True

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
                    x0_predicted, path, nfe, pred_x0, sigmas, _ = karras_sample(
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
                val_loss = L1Loss()(x0_predicted,batch)
                val_loss_step += val_loss.item()
                # print(f'Validation epoch {idx}: {val_loss_step/(idx+1)} \n size batch : {val_batch.shape[0]}')
    
            val_loss_step /= len(self.valdata)
            print(f'Validation loss cohort: {val_loss_step}')
            # self.ddp_model.train()
            
            return val_loss_step

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            batch = batch.to(self.device, dtype=torch.float32) #output
            cond = {
                k: (v.to(self.device, dtype=torch.float32) if v is not None else None)
                for k, v in cond.items()
            }

            t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_bridge_losses, 
                self.ddp_model,
                batch, 
                t, 
                loss_type=self.loss_type,
                model_kwargs=cond,
                bsz_orig = self.batch_size,
                n_slices=self.n_slices_persand,
                continuity=True,
                c_weight=self.continuity_weight,
                )

            loss, loss_img, loss_grad, loss_continuity, _ = compute_losses()

            if torch.isnan(loss):
                loss = torch.tensor(self.loss_acc).to(self.device) # just to keep stats nice
            else:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                self.scaler.step(self.opt)
                self.scaler.update()

            self.cumul_loss += loss.item()
            self.contin_loss += loss_continuity.item()
            self.loss_acc = self.cumul_loss / (self.step + 1)
            self.loss_cont_acc = self.contin_loss / (self.step + 1)

        print('   Iteration ' + str(self.step + 1) + ', loss = ' + str(self.loss_acc), end="\r")

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.ddp_model.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self, for_preemption=False):
        logger.log(f"saving model...")
        filename = f"opt.pt"
        with bf.BlobFile(
            bf.join(get_blob_logdir(), filename),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)

        model_state_dict = self.ddp_model.state_dict()

        checkpoint_data = {"model": model_state_dict,
                            "ema": self.ema.state_dict(),
                            "loss": self.loss_acc,
                            "step": self.step+self.last_step,
                        }
        with bf.BlobFile(bf.join(get_blob_logdir(), f"ckpt_s{self.step+self.last_step}.pt"), "wb") as f:
            torch.save(checkpoint_data, f)

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split("/")[-1].startswith("freq"):
        prefix = "freq_"
    else:
        prefix = ""
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None