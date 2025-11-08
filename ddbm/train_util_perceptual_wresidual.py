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
from ddbm.karras_diffusion import karras_sample_wresidual
from . import logger
from .nn import update_ema, requires_grad
from copy import deepcopy
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
        validate_every,
        test_interval, #how often do you compute the test 
        save_interval, # how often do you save the ckpt !!! IMPORTANT $SAVE_ITER
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        c_weight=0.0,
        p_weight=0.0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=500000, #TODO
        augment_pipe=None,
        train_mode="ddbm",
        resume_train_flag=False,
        n_slices_persand=1,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.valdata=val_data
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
        self.perceptual_weight = p_weight
        self.validate_every=validate_every
        self.min_validation=0.001
        self.patience_counter=0
        self.increasing_validation=0
        self.already_validated=False
        

        if 'perceptual' in self.loss_type:
            self.perceptual = True

        print(f'Training loop running for {self.total_training_steps} steps')
        self.ddp_model = self.model
        self.device='cuda:0'
    
        self.opt = RAdam(self.ddp_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train_mode = train_mode
        self.step = 0
        self.last_step = self.step
        self.resume_train_flag = resume_train_flag
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        self.cumul_loss = 0.0
        self.cumul_percept = 0.0
        self.accum = 2
        
        self._load_and_sync_parameters()
        self._load_optimizer_state()
        self.ema = copy.deepcopy(self.ddp_model).to(self.device)
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
            self.loss_acc = torch.load(resume_checkpoint, weights_only=True)["loss"]
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
            self.ddp_model.train()
            for batch in self.data:
                inputs, gradient_imgs, dists, outputs = [torch.stack(obj, dim=0).squeeze() for obj in batch]
                inputs = inputs.reshape(-1,2,160,160) # bsz, 2, 160, 160
                outputs = outputs.reshape(-1,160,160).unsqueeze(1) # bsz, 1, 160, 160
                dists = dists.reshape(-1,2) # bsz, 2
                gradient_imgs = gradient_imgs.reshape(-1,160,160).unsqueeze(1) # bsz, 1, 160, 160
                dists_th = dists[..., None, None].repeat(1, 1, *self.siz)
                w1 = dists_th[:, 1] / (dists_th.sum(dim=1))
                w2 = 1 - w1
                xT = (w1 * inputs[:, 0] + w2 * inputs[:, 1]).unsqueeze(1) # shape bsz, 160, 160
                residual = outputs - xT
                slab = torch.concat([inputs, self.dist_scale * dists_th], dim=1)

                if self.step >= self.total_training_steps: 
                    print('MAX # TRAINING STEPS REACHED, TERMINATING LOOP')
                    return False
                
                cond = {"xT": xT, "slab" : slab, "dists": dists*self.dist_scale, "gradient_gt": gradient_imgs}

                took_step = self.run_step(residual, outputs, cond)
                
                if took_step and self.step % self.log_interval == 0:
                    wandb.log({'train/loss': self.loss_acc}, step=self.step+self.last_step)
                    self.validate_step(outputs, cond)
                    wandb.log({'train/loss_perceptual': self.loss_percept}, step=self.step+self.last_step)

                if took_step and self.step % self.save_interval == 0: #if took_step is True and we're in save_interval mode
                    self.save() 
                    
                    # torch.cuda.empty_cache()

    def run_step(self, residual, batch, cond): #training step of the validation data
        self.forward_backward(residual, batch, cond) # inputs: x0, xT
        self.step += 1
        update_ema(self.ema, self.model)
        self._anneal_lr()
        return True


    def forward_backward(self, residual, batch, cond):
        self.opt.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            batch = batch.to(self.device, dtype=torch.float16) #output
            residual = residual.to(self.device, dtype=torch.float16) #output
            cond = {
                k: (v.to(self.device, dtype=torch.float16) if v is not None else None)
                for k, v in cond.items()
            }
            t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_bridge_losses_wresidual, 
                self.ddp_model,
                batch, 
                residual,
                t, 
                loss_type=self.loss_type,
                model_kwargs=cond,
                bsz_orig = self.batch_size,
                n_slices=self.n_slices_persand,
                c_weight=self.continuity_weight,
                p_weight=self.perceptual_weight,
                perceptual=self.perceptual,
                )

            loss, loss_img, loss_grad, loss_continuity, loss_perceptual = compute_losses()

            if torch.isnan(loss):
                loss = torch.tensor(self.loss_acc).to(self.device) # just to keep stats nice
            else:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                self.scaler.step(self.opt)
                self.scaler.update()
                
            self.cumul_loss += loss.item()
            self.cumul_percept += loss_perceptual.item()
            self.loss_acc = self.cumul_loss / (self.step + 1)
            self.loss_percept = self.cumul_percept / (self.step + 1)
            print(f"   Iteration {self.step + 1}, loss = {self.loss_acc:.3f}", end="\r")
    
    def validate_step(self, outputs, cond):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                cond = {
                    k: (v.to(self.device) if v is not None else None)
                    for k, v in cond.items()
                }

                x0_predicted, residual_predicted, path, nfe, pred_x0, sigmas, _ = karras_sample_wresidual(self.diffusion, self.ddp_model, cond["xT"], None, steps=10,mask=None,model_kwargs=cond,device=self.device,clip_denoised=True,sampler="heun",churn_step_ratio=0.0,eta=0.0,order=1,seed=42,)
                inputs = cond["slab"][0,:2].cpu()
                distances = cond["slab"][0,2:,0,0].cpu()
                outputs = outputs[0,0].cpu()
                prediction = x0_predicted[0,0].cpu()
                residual = residual_predicted[0,0].cpu()
                interpolation = cond["xT"][0,0].cpu()

                fig = plt.figure()
                plt.subplot(3,3,1)
                plt.imshow(inputs[0], cmap='gray')
                plt.title(f"d1:{distances[0]:.3f}")
                plt.subplot(3,3,2)
                plt.imshow(inputs[1], cmap='gray')
                plt.title(f"d2:{distances[1]:.3f}")
                plt.subplot(3,3,3)
                plt.imshow(prediction, cmap='gray')
                plt.title('Pred x0')
                plt.subplot(3,3,4)
                plt.imshow(outputs, cmap='gray')
                plt.title('GT x0')
                plt.subplot(3,3,5)
                plt.imshow(residual, cmap='gray')
                plt.title('res')
                plt.subplot(3,3,6)
                plt.imshow(interpolation, cmap='gray')
                plt.title('xT')
                fig.savefig(f'{get_blob_logdir()}/sample_{self.step}.png')
                plt.close()

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
                            "opt": self.opt.state_dict()
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