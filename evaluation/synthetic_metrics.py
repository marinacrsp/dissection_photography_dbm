import torch
import os
import math
import logging
import argparse
from CADD.datasets.photosynth_v2 import Photosynth
from CADD.datasets.photo_utils import *
from pathlib import Path
import cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.nn import L1Loss
from CADD.datasets.photosynth_v2 import make_affine_matrix
import csv
from openpyxl import Workbook
from CADD.models.perceptual import PerceptualLoss
from CADD.models.ssim import SSIM3D
from CADD.models.pnsr import PSNRMetric
import json
import pandas as pd
from ddbm.script_util import (
    add_dict_to_argparser,
    args_to_dict,
)

psnr = PSNRMetric()


def custom_psnr(vol1, vol2):
    assert len(vol1) == len(vol2) 

    score_ax0 = []
    score_ax1 = []
    score_ax2 = []

    for i in range(vol2.shape[0]): #sagittal
        score_ax0.append(psnr.calculate_psnr(vol2[i].unsqueeze(0).unsqueeze(0), vol1[i].unsqueeze(0).unsqueeze(0)))
    score_ax0 = torch.stack(score_ax0).mean()

    for i in range(vol2.shape[2]): # axial 
        score_ax1.append(psnr.calculate_psnr(vol2[..., i].unsqueeze(0).unsqueeze(0), vol1[... ,i].unsqueeze(0).unsqueeze(0)))
    score_ax1 = torch.stack(score_ax1).mean()

    for i in range(vol2.shape[1]): # coronal
        score_ax2.append(psnr.calculate_psnr(vol2[:, i].unsqueeze(0).unsqueeze(0), vol1[:, i].unsqueeze(0).unsqueeze(0)))
    score_ax2 = torch.stack(score_ax2).mean()

    all_scores = torch.stack([score_ax0, score_ax1, score_ax2])

    return all_scores, all_scores.mean()

def create_argparser():
    defaults = dict(
        folder="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    root_folder = Path(args.folder)
    device = "cuda:0"
    wb = Workbook()
    wb.remove(wb.active)  # remove the default sheet

    lpips = PerceptualLoss(dimensions=3).to(device)
    ssim = SSIM3D().to(device)

    subfolders = [f for f in root_folder.iterdir() if f.is_dir() and f.name.startswith("ckpt_")]
    gt_filenames = [f for f in root_folder.iterdir() if not f.is_dir() and f.name.startswith(("imputed", "synth_photo_recon"))]
    unet_path=f"{root_folder}/imputed_unet_synth.mgz"
    gt_file = f"{root_folder}/synthetic_gt.nii.gz"
    unet_volume,_ = MRIread(unet_path)
    gt_vol,_ = MRIread(gt_file)
    gt_vol, unet_vol = torch.tensor(gt_vol, dtype=torch.float, device=device), torch.tensor(unet_volume, dtype=torch.float,  device=device)
    _, psnr = custom_psnr(gt_vol, unet_vol)
    _3dssim = ssim.forward(gt_vol.unsqueeze(0).unsqueeze(0), unet_vol.unsqueeze(0).unsqueeze(0))
    perceptual = lpips(gt_vol.unsqueeze(0).unsqueeze(0), unet_vol.unsqueeze(0).unsqueeze(0))    
    

    labels = [str(i) for i in [2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 26]]
    extra = ["mean", "std", "weighted-mean", "weighted-subcortical-mean"]
    all_labels = labels + extra
    df_out = pd.DataFrame({"Label": all_labels})


    for file_name in gt_filenames:
        
        if ".mgz" in str(file_name):
            seg_name = str(file_name).replace(".mgz", "")
        elif ".nii.gz" in str(file_name):
            seg_name = str(file_name).replace(".nii.gz", "")

        if "unet" in seg_name:
            method="unet"
        elif "photo_recon" in seg_name:
            method = "photo_recon"

        seg_file = f"{seg_name}_segmentation/segmentation_score.json"

        with open(seg_file, 'r') as f:
            data = json.load(f)
        dice_data = data["measures"]["dice"]
        jaccard_data= data["measures"]["jaccard"]
        dice_vals = [dice_data["labels"].get(k, dice_data.get(k, None)) for k in all_labels]
        jaccard_vals = [jaccard_data["labels"].get(k, jaccard_data.get(k, None)) for k in all_labels]

        df_out[f"{method}_Dice"] = dice_vals
        df_out[f"{method}_Jaccard"] = jaccard_vals


    with pd.ExcelWriter(root_folder / "segmentation_scores_comparison.xlsx") as writer:
        df_out.to_excel(writer, index=False, sheet_name="SegmentationScores")

    target_files = {
        "heun_vanilla_10st.mgz": "Vanilla",
        "heun_continuity_vanilla_10st.mgz": "Continuity",
        "heun_perceptual_vanilla_10st.mgz": "Perceptual",
        "heun_dist_cond_10st.mgz": "DistCond",
    }

    for subdir in subfolders:
        ws = wb.create_sheet(title=subdir.name[:31])
        ws.append(["File", "3dssim", "psnr", "lpips"])
        for fname, method in target_files.items():

            pred_path = f"{subdir}/{fname}"
            interp_dbi, _ = MRIread(pred_path)
            dbi_vol = torch.tensor(interp_dbi, dtype=torch.float, device=device)
            _, psnr = custom_psnr(gt_vol, dbi_vol)
            _3dssim = ssim.forward(gt_vol.unsqueeze(0).unsqueeze(0), dbi_vol.unsqueeze(0).unsqueeze(0))
            perceptual = lpips(gt_vol.unsqueeze(0).unsqueeze(0), dbi_vol.unsqueeze(0).unsqueeze(0))    
            ws.append([fname, _3dssim.item(), psnr.item(), perceptual.item()])

            ## find segmentation folders:
            seg_name = fname.replace("_10st.mgz", "")
            seg_file = subdir / f"{seg_name}_segmentation/segmentation_score.json"
            with open(seg_file, 'r') as f:
                data = json.load(f)
            dice_data = data["measures"]["dice"]
            jaccard_data= data["measures"]["jaccard"]
            dice_vals = [dice_data["labels"].get(k, dice_data.get(k, None)) for k in all_labels]
            jaccard_vals = [jaccard_data["labels"].get(k, jaccard_data.get(k, None)) for k in all_labels]

            df_out[f"{method}_Dice"] = dice_vals
            df_out[f"{method}_Jaccard"] = jaccard_vals

        with pd.ExcelWriter(subdir / "segmentation_scores_comparison.xlsx") as writer:
            df_out.to_excel(writer, index=False, sheet_name="SegmentationScores")


    wb.save(root_folder / "comparison_metrics.xlsx")
    print(f"Saved comparison_metrics for file {args.folder}")
if __name__ == "__main__":
    main()
    # Save to Excel
