import torch
import os
import math
import logging
import argparse
from CADD.datasets.photo_utils import *
from pathlib import Path
import cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.nn import L1Loss
import csv
from openpyxl import Workbook
from CADD.models.perceptual import PerceptualLoss
from CADD.models.ssim import SSIM3D
from CADD.models.pnsr import PSNRMetric
import json
import re
import pandas as pd
from ddbm.script_util import (
    add_dict_to_argparser,
    args_to_dict,
)
import matplotlib.pyplot as plt
psnr = PSNRMetric()

def mean_finite_psnr(vol_a, vol_b, dim):
    """
    Compute mean PSNR along a given dimension,
    ignoring infinite values.
    """
    # Move the chosen axis to position 0 (batch dimension)
    a = vol_a.movedim(dim, 0)
    b = vol_b.movedim(dim, 0)

    # Add channel & batch dimensions: (N, 1, H, W)
    a = a.unsqueeze(1)
    b = b.unsqueeze(1)

    # Compute PSNR for all slices at once
    psnr_vals = psnr.calculate_psnr(a, b)  # Should return (N,) or (N,1) tensor

    # Filter finite values
    psnr_vals = psnr_vals[torch.isfinite(psnr_vals)]

    return psnr_vals.mean()

def create_argparser():
    defaults = dict(
        folder="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def custom_psnr(vol1, vol2):
    score_ax0 = []
    score_ax1 = []
    score_ax2 = []
    for i in range(vol2.shape[0]): #sagittal
        psnr_value = psnr.calculate_psnr(vol2[i].unsqueeze(0).unsqueeze(0), vol1[i].unsqueeze(0).unsqueeze(0))
        if torch.is_tensor(psnr_value) and torch.isfinite(psnr_value):
            score_ax0.append(psnr_value)
    # score_ax0 = score_ax0[np.isfinite(score_ax0.cpu().numpy())]
    score_ax0 = torch.stack(score_ax0).mean()

    for i in range(vol2.shape[2]): # axial 
        psnr_value = psnr.calculate_psnr(vol2[..., i].unsqueeze(0).unsqueeze(0), vol1[... ,i].unsqueeze(0).unsqueeze(0))
        if torch.is_tensor(psnr_value) and torch.isfinite(psnr_value):
            score_ax1.append(psnr_value)
    score_ax1 = torch.stack(score_ax1).mean()

    for i in range(vol2.shape[1]): # coronal
        psnr_value = psnr.calculate_psnr(vol2[:, i].unsqueeze(0).unsqueeze(0), vol1[:, i].unsqueeze(0).unsqueeze(0))
        if torch.is_tensor(psnr_value) and torch.isfinite(psnr_value):
            score_ax2.append(psnr_value)
    score_ax2 = torch.stack(score_ax2).mean()

    all_scores = torch.stack([score_ax0, score_ax1, score_ax2])
    return all_scores.mean()


class Get_voxel_metrics():

    def __init__(self,
                 device
                 ):
        
        self.lpips = PerceptualLoss(dimensions=3).to(device)
        self.ssim = SSIM3D().to(device)
        self.psnr = PSNRMetric()

    def __call__(self, gt_vol, vol_tensor):
        # Dummy metric values — replace with your actual computation

        psnr =self.psnr.calculate_psnr(gt_vol, vol_tensor)
        _3dssim = self.ssim(gt_vol.unsqueeze(0).unsqueeze(0), vol_tensor.unsqueeze(0).unsqueeze(0))
        perceptual_score = self.lpips(gt_vol.unsqueeze(0).unsqueeze(0), vol_tensor.unsqueeze(0).unsqueeze(0))

        return {
            "PSNR": psnr.item(),
            "LPIPS": perceptual_score.item(),
            "SSIM": _3dssim.item()
        }


def main():
    args = create_argparser().parse_args()
    good_files = {"interp_synthvol_00","interp_synthvol_03","interp_synthvol_05","interp_synthvol_07","interp_synthvol_010","interp_synthvol_011","interp_synthvol_012","interp_synthvol_015","interp_synthvol_016","interp_synthvol_017","interp_synthvol_019","interp_synthvol_021","interp_synthvol_023","interp_synthvol_030","interp_synthvol_031","interp_synthvol_032","interp_synthvol_034","interp_synthvol_036","interp_synthvol_038","interp_synthvol_039","interp_synthvol_040","interp_synthvol_041","interp_synthvol_042","interp_synthvol_043","interp_synthvol_044","interp_synthvol_045","interp_synthvol_046","interp_synthvol_047","interp_synthvol_048","interp_synthvol_055","interp_synthvol_057","interp_synthvol_058","interp_synthvol_061","interp_synthvol_064","interp_synthvol_065","interp_synthvol_075","interp_synthvol_078","interp_synthvol_079","interp_synthvol_080","interp_synthvol_084","interp_synthvol_085","interp_synthvol_088","interp_synthvol_089","interp_synthvol_090","interp_synthvol_094","interp_synthvol_096","interp_synthvol_098","interp_synthvol_099"}

    root_folder = Path(args.folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wb = Workbook()
    wb.remove(wb.active)  # remove the default sheet
    get_metrics_volume = Get_voxel_metrics(device)
    lpips = PerceptualLoss(dimensions=3).to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PSNRMetric()
    identifiers = ['5', '8', '15', '18', '20']
    for file in good_files:
    
        scan_path = os.path.join(root_folder, f"{file}")
        gt_name = f"{scan_path}/synthetic_gt.nii.gz"
        gt_vol,_= MRIread(gt_name)

        mask_nonzeros = np.array(gt_vol>0.05).astype(np.uint8) ## Background
        
        gt_vol=torch.tensor(gt_vol, dtype=torch.float, device=device)
        for dist in identifiers:
            dist_folder = os.path.join(scan_path, f"dist_{dist}")
            if not os.path.isdir(dist_folder):
                continue

            voxel_metrics = {}

            for fname in os.listdir(dist_folder):
                fpath = os.path.join(dist_folder, fname)   

                if not fname.startswith("synth_photo_recon"):
                    if fname.endswith(".nii.gz") or fname.endswith(".mgz"):
                        fvol,_ = MRIread(str(fpath))

                        if fname.startswith("imputed_unet") and 'resampled' in fname:
                            method = "unet"
                        elif fname.startswith("imputed_dbi") and 'resampled' in fname:
                            try:
                                m = re.match(r"imputed_dbi(.+)\.mgz", fname)
                                method = m.group(1)  # e.g. unet_mse_heun_vp
                            except:
                                breakpoint()
                        else: 
                            continue
                        
                        fvol = (fvol - fvol.min()) / (fvol.max() - fvol.min()) 
                        gt_vol = (gt_vol - gt_vol.min()) / (gt_vol.max() - gt_vol.min()) 
                        fvol = fvol*mask_nonzeros
                        fvol = torch.tensor(fvol, dtype=torch.float32, device=device)

                        metrics = get_metrics_volume(gt_vol, fvol)
                        voxel_metrics[method] = metrics
                    
            # Save to Excel
            voxel_df = pd.DataFrame(voxel_metrics).sort_index()
            excel_path = os.path.join(dist_folder, f"voxel_metrics_{dist}.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                voxel_df.to_excel(writer, sheet_name="voxel_based")
                # Leave volume_based blank for now
            print(f"Saved voxel metrics to {excel_path}")

if __name__ == "__main__":
    main()
    # Save to Excel
