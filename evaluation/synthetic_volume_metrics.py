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

psnr = PSNRMetric()


def create_argparser():
    defaults = dict(
        folder="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


class Get_vol_metrics():

    def __init__(self,
                 device
                 ):
        
        self.lpips = PerceptualLoss(dimensions=3).to(device)
        self.ssim = SSIM3D().to(device)
        self.psnr = PSNRMetric()
        self.labels = [2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 26, 77, 819, 843, 865, 869]

    def __call__(self, 
                 gt_vol, 
                 vol_tensor):
        """
        Inputs are both numpy arrays
        """

        dice_scores = {}
        jaccard_scores = {}
        vol_errors = {}

        for label in self.labels:
            pred_l = (vol_tensor == label).astype(np.uint8) # convert to binary
            gt_l = (gt_vol == label).astype(np.uint8) # convert to binary

            intersection = np.sum(pred_l * gt_l)
            pred_sum = np.sum(pred_l)
            gt_sum = np.sum(gt_l)
            dice = (2 * intersection / (pred_sum + gt_sum + 1e-8))
            union = pred_sum + gt_sum - intersection
            jaccard = (intersection / (union + 1.e-8))
            ve = abs(float(pred_sum) - float(gt_sum)) / (gt_sum + 1e-8)

            dice_scores[label] = dice.item()
            jaccard_scores[label] = jaccard.item()
            vol_errors[label] = ve.item()

        return {
            "DICE": dice_scores,
            "JACCARD": jaccard_scores,
            "VOL": vol_errors
        }

def format_metric_table(metric_dict, label_counts, metric_name):
    df = pd.DataFrame(metric_dict).sort_index()  # rows: label, cols: methods
    df.index.name = "Label"

    # Stats
    mean_row = df.mean(axis=0)
    weighted_row = df.apply(
        lambda col: np.average(col, weights=[label_counts.get(l, 0) for l in df.index]), axis=0
    )
    std_row = df.std(axis=0)

    df.loc[f"{metric_name}_mean"] = mean_row
    df.loc[f"{metric_name}_weighted_mean"] = weighted_row
    df.loc[f"{metric_name}_std"] = std_row
    return df

def main():
    args = create_argparser().parse_args()
    root_folder = Path(args.folder)
    device = "cuda:0"
    wb = Workbook()
    wb.remove(wb.active)  # remove the default sheet
    get_metrics_volume = Get_vol_metrics(device)

    identifiers = ['10']

    for i in range(0, 51, 1):
        scan_path = os.path.join(root_folder, f"interp_synthvol_0{i}")
        gt_name = f"{scan_path}/synthetic_gt.nii.gz"
        seg_gt=f"{scan_path}/synthetic_gt_segmentation/segmentation.mgz"
        seg_gtvol,_= MRIread(seg_gt)

        for dist in identifiers:
            dist_folder = os.path.join(scan_path, f"dist_{dist}-best")
            if not os.path.isdir(dist_folder):
                continue
            
            volume_metrics = {}

            for fname in os.listdir(dist_folder):
                fpath = os.path.join(dist_folder, fname) 
                seg_path = os.path.join(fpath, "segmentation.mgz")  
                
                if os.path.isfile(seg_path):
                    if "synth_photo_recon" in fpath:
                        method = "photo_recon"
                    elif "segmentation_imputed_unet" in fpath:
                        method = "unet"
                    # elif "segmentation_imputed_inr" in fpath:
                    #     method = "inr"
                    elif "segmentation_imputed_dbi_" in fpath:
                        method = fpath[fpath.find("segmentation_imputed_dbi_"):]
                        method = method.replace("segmentation_imputed_dbi_", "")
                    else: 
                        continue
                    seg_vol,_ = MRIread(str(seg_path))
                    slicediff = abs(seg_gtvol.shape[1] - seg_vol.shape[1])
                    if slicediff != 0:
                        if slicediff % 2 == 0:  # even!!!
                            seg_vol = seg_vol[:,slicediff//2:-slicediff//2]
                        else:
                            continue

                    metrics = get_metrics_volume(seg_gtvol, seg_vol)
                    volume_metrics[method] = metrics

            dice_table = {}
            jaccard_table = {}
            roi_table = {}
            label_counts = {}

            for method, result in volume_metrics.items():
                dice_table[method] = result["DICE"]
                jaccard_table[method] = result["JACCARD"]
                roi_table[method] = result["VOL"]
                if not label_counts:
                    label_counts = {label: np.sum(seg_gtvol == label) for label in get_metrics_volume.labels}

            excel_path = os.path.join(dist_folder, f"volume_metrics_{dist}.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                format_metric_table(dice_table, label_counts, "DICE").to_excel(writer, sheet_name="Dice")
                format_metric_table(jaccard_table, label_counts, "JACCARD").to_excel(writer, sheet_name="Jaccard")
                format_metric_table(roi_table, label_counts, "ROI").to_excel(writer, sheet_name="VolumeError")

            print(f"Saved metrics to {excel_path}")

if __name__ == "__main__":
    main()
    # Save to Excel
