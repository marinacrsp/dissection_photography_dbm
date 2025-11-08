import torch
import os
import math
import logging
import argparse
from CADD.models.ssim import SSIM3D
from CADD.models.pnsr import PSNRMetric
from datasets.photo_utils import *
import pandas as pd


def main():
    indir="/cluster/vive/UW_photo_recon/FLAIR_Scan_Data/Rotated_QC/UWA_fixed_tissue_dataset_bids/derivatives/12_photo_recons_4_marina/"
    device='cuda'

    ssim = SSIM3D().to(device)
    psnr = PSNRMetric()
    synthetic_volumes = ["SynthT1", "SynthT2", "SynthFLAIR", "fakeCortex"]
    metrics = ["SSIM", "PSNR"]

    for file in os.listdir(indir):
        for dist in ["4mm", "8mm", "12mm"]:
            for method in ["photo_recon", "imputed_unet", "imputed_dbi-L1_gradient_perceptual-vanilla-vp-24"]:
                output_xl = f"{indir}{file}/neurosynth/neurosynth_{method}_{dist}/voxel_scores.xlsx" 
                with pd.ExcelWriter(output_xl, engine="xlsxwriter") as writer:
                    voxel_metrics = pd.DataFrame(index=metrics, columns=synthetic_volumes, dtype=float)
                    for synthvol in synthetic_volumes:
                        vol_pred,_= MRIread(f"/cluster/vive/UW_photo_recon/FLAIR_Scan_Data/Rotated_QC/UWA_fixed_tissue_dataset_bids/derivatives/12_photo_recons_4_marina/19-0019/neurosynth/neurosynth_{method}_{dist}/{synthvol}.mgz")
                        vol_mri,_= MRIread(f"/cluster/vive/UW_photo_recon/FLAIR_Scan_Data/Rotated_QC/UWA_fixed_tissue_dataset_bids/derivatives/12_photo_recons_4_marina/19-0019/neurosynth/neurosynth_mri/{synthvol}.mgz")
                        
                        mask = vol_mri
                        ## Normalization to 0,1 for pixel-based comparison
                        vol_pred = (vol_pred - vol_pred.min())/(vol_pred.max() - vol_pred.min())
                        vol_mri = (vol_mri - vol_mri.min())/(vol_mri.max() - vol_mri.min())

                        vol_pred, vol_mri = torch.tensor(vol_pred, device=device), torch.tensor(vol_mri, device=device)
                        ssim_score, psnr_score = ssim(vol_mri.unsqueeze(0).unsqueeze(0), vol_pred.unsqueeze(0).unsqueeze(0)),  psnr.calculate_psnr(vol_mri, vol_pred)
                        voxel_metrics[synthvol]["SSIM"] = float(ssim_score)
                        voxel_metrics[synthvol]["PSNR"] = float(psnr_score)

                    voxel_metrics.to_excel(writer, float_format="%.3e")

                    print(f"Saved file as {output_xl}")

if __name__ == "__main__":
    main()
