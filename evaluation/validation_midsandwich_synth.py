import sys
import os
import argparse
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from datetime import datetime
import matplotlib.pyplot as plt
from ddbm.karras_diffusion import karras_sample
import torch
import numpy as np
from omegaconf import OmegaConf
import cv2
import csv
from ddbm.nn import mean_flat
from ddbm import dist_util, logger
import torch.nn.functional as F
from CADD.models.perceptual import PerceptualLoss
from CADD.models.pnsr import PSNRMetric
from pathlib import Path
from torchmetrics.image import StructuralSimilarityIndexMeasure
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torch.nn import L1Loss
import pandas as pd


def ssim(input1, input2):
    return StructuralSimilarityIndexMeasure()(input1.unsqueeze(0).unsqueeze(0), input2.unsqueeze(0).unsqueeze(0))
def mse(input1, input2):
    return mean_flat((input1 - input2)**2).mean()
def l1loss(input1, input2):
    return L1Loss()(input1,input2)
def correlation(input1, input2):
    num = (input1 * input2).sum()
    denom = ((input1**2).sum() * (input2**2).sum()).sqrt()
    return num/denom if denom != 0 else 0
def perceptual_loss(input1, input2):
    return PerceptualLoss(dimensions=2)(input1.unsqueeze(0).unsqueeze(0), input2.unsqueeze(0).unsqueeze(0))
def compute_psnr(input1, input2):
    return PSNRMetric().calculate_psnr(input1.unsqueeze(0).unsqueeze(0), input2.unsqueeze(0).unsqueeze(0))


def create_argparser():
    defaults = dict(
        data_dir="",  ## only used in bridge
        dataset="edges2handbags",
        clip_denoised=True,
        num_samples=10000,
        batchsize=4,
        sampler="heun",
        split="train",
        churn_step_ratio=0.0,
        rho=7.0,
        steps=10,
        model_path="",
        exp="",
        seed=42,
        num_workers=4,
        eta=1.0,
        order=1,
        save_path="",
        gt_file="",
        illumination=None,
        unsharp_sigma=1.0,
        unsharp_amount=1.0,
        use_dist_conditioning=False,
        unet_type="adm",
    )
    
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def visualize(j, s1, s2, pred, gt, d1, d2,sample_dir):
    s1 = s1.squeeze().detach().cpu(); s2 = s2.squeeze().detach().cpu(); pred = (pred).squeeze().detach().cpu()
    ssim_thisj = ssim(pred, gt)
    psnr_thisj = compute_psnr(pred, gt)

    lpip_thisj = perceptual_loss(pred, gt)
    mse_thisj = mse(pred, gt)
    l1_loss_thisj = l1loss(pred, gt)
    # residue = (pred.squeeze().numpy()-gt)**2
    if j % 10 ==0:
        plt.figure(figsize=(10,5)); 
        plt.subplot(2,2,1); 
        plt.imshow(s1, cmap='gray'); 
        plt.title('Input1'); 
        plt.axis('off'); 
        plt.subplot(2,2,4); 
        plt.imshow(gt, cmap='gray'); 
        plt.title('gt'); plt.axis('off'); 
        plt.subplot(2,2,3); 
        plt.imshow(pred, cmap='gray'); 
        plt.axis('off'); 
        plt.title(f'ssim: {ssim_thisj:.3f} \n mse: {mse_thisj:.3f} \n l1loss: {l1_loss_thisj:.3f}  \n ncc: {psnr_thisj:.3f}'); 
        plt.subplot(2,2,2); 
        plt.imshow(s2, cmap='gray'); 
        plt.title('Input2'); plt.axis('off'); 
        plt.tight_layout(); 
        plt.tight_layout(); 
        plt.savefig(f'{sample_dir}/dist_{(d1 + d2).item():.3f}.png'); 
        plt.close()
    return ssim_thisj, mse_thisj, l1_loss_thisj, lpip_thisj, psnr_thisj
def main():
    args = create_argparser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.use_fp16 = False
    dtype=torch.float32
    dist_scale = 0.1
    model_path = args.model_path
    # model_path="/autofs/space/almeria_001/users/marina/dbi/workdir/dbi-L1_gradient/model_ckpt.pt"
    workdir = model_path[model_path.find("workdir"):-3]
    sample_dir = Path(workdir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    ckpt=torch.load(model_path,
                     weights_only=True)["ema"]

    if "i2sb" in args.model_path:
        scheduler = "i2sb"
    else:
        scheduler = "vp"
    args.noise_schedule=scheduler
    model, diffusion = create_model_and_diffusion( # initialize the karras_denoiser
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    if args.unet_type == "vanilla":
        for name, param in model.named_parameters():
            param.data.copy_(ckpt[name])
    
    else:
        model.load_state_dict(ckpt)

    name_batch = "/homes/1/ma1282/marina_almeria/dbi/workdir/validation_normalized_stack.pth"
    batch_synth = torch.load(name_batch)
    batch_synth = torch.tensor(batch_synth, dtype=dtype) # the batch of sandwiches generated
    metrics_name = name_batch[name_batch.find('validation_normalized_stack'): name_batch.find('.pth')]
    
    model = model.to(device)
    d1,d2 = batch_synth[:,2,0,0], batch_synth[:,3,0,0]
    input_batch = batch_synth[:,:-1].clone() 
    input_batch[:,2:] *= dist_scale # distannce

    w1 = (d2/(d1+d2)).unsqueeze(-1).unsqueeze(-1).repeat(1,*input_batch.shape[2:])
    w2 = 1- w1
    linear_interp = input_batch[:,0]*w1 + input_batch[:,1]*w2

    model.eval()

    diff_batch = torch.cat([input_batch, linear_interp.unsqueeze(1)], dim=1)
    outputs = torch.split(batch_synth[:,-1], 64)
    dists = torch.split(batch_synth[:,2:-1], 64)
    diff_batch_split =  torch.split(diff_batch, 64)

    ssim_score = []
    mse_score = []
    l1_score = []
    psnr_score = []
    lpips_score=[]
    abs_dist =[]
    with torch.no_grad():
    
        for i, batch in enumerate(diff_batch_split):
            batch = batch.to(device, dtype=dtype)
            outputs_b = outputs[i].to(device)
            d1, d2 = dists[i][:,0,0,0].unsqueeze(1), dists[i][:,1,0,0].unsqueeze(1)
            dists_model = torch.cat([d1, d2], dim=1)
            model_kwargs = {"xT": torch.tensor(batch[:,-1].unsqueeze(1), dtype=dtype), 
                        "slab": torch.tensor(batch[:,:-1], dtype=dtype), 
                        "dists": torch.tensor(dists_model*dist_scale, 
                                                device=device, 
                                                dtype=dtype)
                        }
            # for eta in eta_values:
                
            x0_predicted, path, nfe, pred_x0, sigmas, _ = karras_sample(
                                                                        diffusion,
                                                                        model,
                                                                        model_kwargs["xT"], #prior distribution
                                                                        None, #x0,
                                                                        steps=args.steps,
                                                                        mask=None,
                                                                        model_kwargs=model_kwargs,
                                                                        device=device,
                                                                        clip_denoised=args.clip_denoised,
                                                                        sampler="heun",
                                                                        churn_step_ratio=args.churn_step_ratio,
                                                                        eta=args.eta,
                                                                        order=args.order,
                                                                        seed=42,
                                                                    )
                # visualize(path[0], x0_predicted.squeeze(), outputs_b.squeeze(), eta)

            x0_predicted = x0_predicted.squeeze().detach().cpu()
            for j in range(x0_predicted.shape[0]):
                j_ssim, j_mse, j_l1loss, j_lpips, j_psnr = visualize(j, batch[j,0], batch[j,1], x0_predicted[j], outputs[i][j], d1[j], d2[j], sample_dir)

                ssim_score.append(j_ssim.item())
                mse_score.append(j_mse.item())
                l1_score.append(j_l1loss.item())
                lpips_score.append(j_lpips.item())
                try:
                    if torch.is_tensor(j_psnr):
                        val = j_psnr.item()
                    else:
                        val = float(j_psnr)
                        if math.isinf(val) or math.isnan(val):
                            val = 50
                except Exception:
                    breakpoint()
                psnr_score.append(val)
                abs_dist.append((d1[j] + d2[j]).item())
    

    df = pd.DataFrame({'abs_dist': abs_dist,'ssim_score': ssim_score, 'mse_score': mse_score, 'l1_score': l1_score, 'lpips_score': lpips_score, 'psnr_score': psnr_score,})
    # Save to CSV
    df.to_csv(f'{sample_dir}/metrics_{metrics_name}.csv', index=False)
if __name__ == "__main__":
    main()