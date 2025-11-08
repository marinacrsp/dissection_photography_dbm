import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from datetime import datetime
import matplotlib.pyplot as plt
# If everything went fine, we import the rest of packages
from ddbm.karras_diffusion import karras_sample
import torch
from CADD.datasets.photo_utils import MRIread, MRIwrite, eugenios_closest_canonical, gaussian_blur_3d
import numpy as np
from omegaconf import OmegaConf
import cv2
import csv
from ddbm import dist_util, logger
from CADD.diffusion import create_diffusion
from CADD.download import find_model
from CADD.models import get_models
import torch.nn.functional as F
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from pathlib import Path

def pad_to_symmetric(tensor):
    """
    Pads a [B, C, H, W] tensor so that H and W become 160 using symmetric (centered) padding.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    max_size =  max(h,w)
    pad_h = max(max_size - h, 0) 
    pad_w = max(max_size - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(tensor, padding, mode='constant', value=0) # pad with zeros


def center_crop(tensor, original_hw):
    """
    Center-crops the last two dimensions of the tensor back to original_hw = (H, W).
    Works for tensors with shape [B, C, H, W].
    """
    orig_h, orig_w = original_hw
    h, w = tensor.shape[-2], tensor.shape[-1]

    start_h = (h - orig_h) // 2
    start_w = (w - orig_w) // 2

    return tensor[..., start_h:start_h+orig_h, start_w:start_w+orig_w]

def visualize(j, s1, s2, pred, gt, d1, d2,sample_dir):
    s1 = s1.cpu(); s2 = s2.cpu(); pred = (pred).cpu()
    residue = (pred.squeeze().numpy()-gt)
    plt.figure(figsize=(10,5)); 
    plt.subplot(2,3,1); 
    plt.imshow(s1, cmap='gray'); 
    plt.title('Input1'); 
    plt.axis('off'); 
    plt.subplot(2,3,2); 
    plt.imshow(gt, cmap='gray'); 
    plt.title('gt'); plt.axis('off'); 
    plt.subplot(2,3,3); 
    plt.imshow(pred, cmap='gray'); 
    plt.axis('off'); 
    plt.title(f'Out.\n d1: {d1}, d2: {d2}'); 
    plt.subplot(2,3,4); 
    plt.imshow(s2, cmap='gray'); 
    plt.title('Input2'); plt.axis('off'); 
    plt.tight_layout(); 
    plt.subplot(2,3,5); 
    plt.imshow(residue, cmap="viridis"); 
    plt.title('error'); plt.axis('off'); plt.colorbar()
    plt.tight_layout(); 
    plt.savefig(f'{sample_dir}/sample_{j}.png'); 
    plt.close()

def create_argparser():
    defaults = dict(
        data_dir="",  ## only used in bridge
        dataset="edges2handbags",
        dist_emb=False,
        rel_center=False,
        combine_dists=False,
        condition_mode="",
        clip_denoised=True,
        num_samples=10000,
        batchsize=4,
        sampler="heun",
        split="train",
        churn_step_ratio=0.0,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=42,
        num_workers=4,
        eta=1.0,
        order=1,
        save_path="",
        input_file="",
        illumination=None,
        unsharp_sigma=1.0,
        unsharp_amount=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

# ================================================================================================
#                                         Main Entrypoint
# ================================================================================================
def main():
    args = create_argparser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.use_fp16 = False
    dtype=torch.float32
    dist_scale = 0.1
    print(args)

    workdir = args.model_path[args.model_path.find("workdir"):-3]
    print(f'Mode of evaluation: {args.split}')

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    # split = args.model_path.replace("_adapted", "").split("_")
    # step = int(split[-1].split(".")[0])

    sample_dir = args.input_file[:args.input_file.find("/synth_photo_recon")]
    device = "cuda:0"
    ckpt_name = args.model_path[args.model_path.find("ckpt"):args.model_path.find(".pt")]

    logger.log("creating model and diffusion...")

    if "i2sb" in args.model_path:
        scheduler = "i2sb"
    else:
        scheduler = "vp"
    args.noise_schedule=scheduler
    print(f'Sampling with {scheduler} scheduler')
    model, diffusion = create_model_and_diffusion( # initialize the karras_denoiser
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    method = Path(args.model_path).parent.name

    # if "perceptual" in args.model_path and "vanilla" in args.model_path:
    #     exp_name="vanilla_perceptual"
    # elif "continuity" in args.model_path and "vanilla" in args.model_path:
    #     exp_name="vanilla_continuity"
    # elif "vanilla" in args.model_path:
    #     exp_name="vanilla"
    # elif "perceptual" in args.model_path and "dist_cond" in args.model_path:
    #     exp_name="dist_cond_perceptual"
    # elif "continuity" in args.model_path and "dist_cond" in args.model_path:
    #     exp_name="dist_cond_continuity"
    # elif "dist_cond" in args.model_path:
    #     exp_name = "dist_cond"

    print(f'Sampling with {args.noise_schedule}')
    # print(f'Sampling EXPERIMENT: {exp_name}')

    ckpt = torch.load(args.model_path, weights_only=True)["model"]
    if args.unet_type == "vanilla":
        for name, param in model.named_parameters():
            param.data.copy_(ckpt[name])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)

    if args.use_fp16:
        model = model.half()
    model.eval()

    logger.log("sampling...")

    with torch.no_grad():
        # Read in input file with reorientation
        print('Reading and resizing input images')
        I_orig, aff_orig = MRIread(args.input_file)
        I_orig, aff_orig, ap_flip = eugenios_closest_canonical(I_orig, aff_orig, return_ap_flip=True)
        if len(I_orig.shape)==3:
            I_orig = I_orig[..., None]

        voxsize = np.sqrt(np.sum(aff_orig[:-1,:-1]**2, axis=0))
        av_thickness = voxsize[1]
        inplane_res = .5 * voxsize[0] + .5 * voxsize[2]

        ## Adapt in-plane resolution to 1mm/px
        I = []
        for j in range(I_orig.shape[1]):
            I.append(cv2.resize(I_orig[:,j,:,:], (0,0), fx=inplane_res, fy=inplane_res, interpolation=cv2.INTER_AREA))
        I = np.stack(I, axis=1)
        aff = aff_orig.copy()
        aff[:-1, 0] /= inplane_res
        aff[:-1, 2] /= inplane_res
        aff[:-1,-1] -= aff[:-1,:-1] @ np.array([0.5*(inplane_res-1), 0, 0.5*(inplane_res-1),])
        
        I = torch.tensor(I, dtype=dtype, device=device)
        
        # convert grayscale to RGB if needed
        if len(I.shape)==3:
            I = I[..., None]
        # Compute mask and normalize by median
        M = I.sum(dim=[3])>0.09

        # Calculate areas and detect padding
        areas = M.sum(dim=[0, 2]).detach().cpu().numpy()
        aux = np.where(areas > 0)
        PAD = aux[0][0].astype(np.int32)

        # if (aux[0][-1] != (I.shape[1] - 4)):
        #     raise Exception('Uneven padding detected in anterior and posterior directions')

        minmax = torch.zeros(I.shape[3], 2, dtype=torch.float32)
        for c in range(I.shape[-1]):
            auxI = I[:,:,:,c]
            minmax[c,0], minmax[c,1] = I[..., c].min(), I[..., c].max()
            I[...,c] =  2*(I[...,c] - auxI.min()) / (auxI.max() - auxI.min()) - 1

        # medians = torch.zeros(I.shape[3], device=arguments.device, dtype=dtype)
        thicknesses = av_thickness * np.ones(M.shape[1] - 2 * PAD + 1)

        # Create output image with appropriate header
        I2 = torch.zeros([I.shape[0], np.ceil(I.shape[1] * av_thickness).astype(np.int32) , I.shape[2], I.shape[3]], device=device, dtype=dtype)
        I2linear = torch.zeros_like(I2)
        aff2 = aff.copy()
        aff2[:-1,1] = aff2[:-1,1] / av_thickness # this is to reshape to 1mm/px
        aff2[:-1,-1] = aff2[:-1,-1] - aff2[:-1,:-1] @ np.array([0, 0.5*(av_thickness-1), 0])

        # Prepare input with size multiple of 32
        shape2d = np.array([I.shape[0], I.shape[2]]) # this is just the original shape of the volume
        W = (np.ceil(shape2d / 32.0) * 32).astype('int') # W is the new shape to be interpolated
        idx = np.floor((W - shape2d) / 2).astype('int')
        slab = torch.zeros([I.shape[3], 5, *W], dtype=torch.float32, device=device)

        # get adjusted coordinates of photos; we assume that the thickness is av_thickness for the empty slices
        y_coords = av_thickness * np.arange(PAD)
        y_coords = np.concatenate([y_coords, y_coords[-1]+thicknesses.cumsum()])
        y_coords = np.concatenate([y_coords, y_coords[-1]+av_thickness * np.arange(1, PAD)])

        batch_slabs = []
        batch_dists=[]

        # Loop over coronal slices and interpolate
        for j in range(I2.shape[1]):
            print('Working on slice ' + str(j+1) + ' of ' + str(I2.shape[1]), end='\r')
            js = j - (av_thickness - 1 ) / 2
            aux = np.where(y_coords<=js)[0]
            idx1 = aux.max() if len(aux)>0 else 0
            idx2 = min(idx1+1, len(y_coords)-1)

            d1 = js - y_coords[idx1]
            d2 = y_coords[idx2] - js

            w1 = (d2 / (d1 + d2)) if ((d1+d2)>0) else 0.0
            w2 = 1 - w1
            linear_interp = w1 * I[:, idx1, :, :] + w2 * I[:, idx2, :, :]
            I2linear[:, j, :, :] = linear_interp
                        
            y0 = linear_interp.to(device).permute(2,0,1).unsqueeze(0)

            slab[0,0, idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2]] = I[:, idx1, :,0]
            slab[0,1, idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2]] = I[:, idx2, :,0]
            slab[0,2, idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2]] = d1 * dist_scale
            slab[0,3, idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2]] = d2 * dist_scale
            slab[0,4, idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2]] = linear_interp.to(dist_util.dev()).permute(2,0,1)[c]
            #pad S so that last two dimensions are 160
            slab = pad_to_symmetric(slab)
            batch_slabs.append(slab.clone())
            batch_dists.append(torch.tensor([dist_scale*d1, dist_scale*d2]))

        batch_slabs = torch.stack(batch_slabs)
        batches_dists = torch.stack(batch_dists, dim=0)
        bsz=20 # since it's singled channeled  
        batches_slabs = torch.split(batch_slabs, bsz)
        batches_dists = torch.split(batches_dists, bsz)
            
        for idx, (ith_batch_slab, ith_batch_dists) in enumerate(zip(batches_slabs, batches_dists)): 
            j_start = idx*(bsz)
            j_end = (idx*bsz + ith_batch_slab.shape[0])
            print('Working on slices ' + str(j_end) + '/' + str(j+1), end='\r')
            this_batch_slabs = ith_batch_slab[:,:,:-1]
            this_batch_slabs = this_batch_slabs.reshape(-1, *this_batch_slabs.shape[2:]).to(device=dist_util.dev())
            xT = ith_batch_slab[:,:,-1]
            xT = xT.reshape(-1, *xT.shape[2:]).unsqueeze(1).to(device=dist_util.dev())
            model_kwargs = {"xT": xT, 
                            "slab": this_batch_slabs, 
                            "dists": ith_batch_dists.to(device=dist_util.dev(), dtype=torch.float32),
                            }

            x0_predicted, path, nfe, pred_x0, sigmas, _ = karras_sample(
                                                                        diffusion,
                                                                        model,
                                                                        xT, #prior distribution
                                                                        None, #x0,
                                                                        steps=args.steps,
                                                                        mask=None,
                                                                        model_kwargs=model_kwargs,
                                                                        device=device,
                                                                        clip_denoised=args.clip_denoised,
                                                                        sampler=args.sampler,
                                                                        churn_step_ratio=args.churn_step_ratio,
                                                                        eta=args.eta,
                                                                        seed=42,
                                                                    )
            x0_predicted = x0_predicted.detach().cpu()

            ### Avoid errors when batch is size 1
            if x0_predicted.shape[0] > 1:
                pred = x0_predicted.squeeze()
            else:
                pred = x0_predicted.squeeze(1)

            #crop pred back to original size
            pred = center_crop(pred, (I2.shape[0], I2.shape[2])).permute(1,0,2).unsqueeze(-1) # last channel is 1
            I2[:, j_start:j_end, :, :] = pred


    # DENORMALIZE
    for c in range(I2.shape[-1]):
        I2[...,c] = (I2[...,c] + 1)*0.5*(minmax[c,1] - minmax[c,0]) + minmax[c,0] # from -1, 1 -> 0, 1
    unsharp_amount, unsharp_sigma = 1, 1
    blurred = torch.zeros_like(I2) 
    
    for c in range(I2.shape[3]):
        blurred[..., c] = gaussian_blur_3d(I2[..., c], [unsharp_sigma, unsharp_sigma, unsharp_sigma], device)      
    I2 += unsharp_amount * (I2 - blurred) # cause in-plane slices look sharper than off-plane slices

    I2 = (I2.squeeze()).clip(0).detach().cpu().numpy()

    #clip
    output_file_nn = f'{sample_dir}/imputed_{method}.mgz'
    MRIwrite(I2, aff2, output_file_nn)

if __name__ == "__main__":
    main()


