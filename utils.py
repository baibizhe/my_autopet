import os.path

import numpy as np
import nibabel as nib
import pathlib as plb
# import cc3d
import csv
import sys

import torch
# from timm.utils.metrics import

def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']
    voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000
    return  voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


# def false_pos_pix(gt_array, pred_array):
#     # compute number of voxels of false positive connected components in prediction mask
#     pred_conn_comp = con_comp(pred_array)
#
#     false_pos = 0
#     for idx in range(1, pred_conn_comp.max() + 1):
#         comp_mask = np.isin(pred_conn_comp, idx)
#         if (comp_mask * gt_array).sum() == 0:
#             false_pos = false_pos + comp_mask.sum()
#     return false_pos


# def false_neg_pix(gt_array, pred_array):
#     # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
#     gt_conn_comp = con_comp(gt_array)
#
#     false_neg = 0
#     for idx in range(1, gt_conn_comp.max() + 1):
#         comp_mask = np.isin(gt_conn_comp, idx)
#         if (comp_mask * pred_array).sum() == 0:
#             false_neg = false_neg + comp_mask.sum()
#
#     return false_neg


def dice_score(mask1, mask2):



    # compute foreground Dice coefficient
    overlap = (mask1 * mask2).sum()
    sum = mask1.sum() + mask2.sum()
    dice_score =( 2 * overlap+1) / (sum+1)
    return dice_score


def compute_metrics(gt_array, pred_array,config):
    # main function
    if len(gt_array.shape)>3:
        batch_size = gt_array.shape[0]
    # assert gt_array.shape == pred_array.shape,"The shape between get_array and pred_array should be same,but got gt shape{} pred shape{}".format(str(gt_array.shape),str(pred_array.shape))
    # nii_path = os.path.join("data","nifti","PETCT_dadcd69ba9","05-19-2005-NA-PET-CT Ganzkoerper  primaer mit KM-61889","SEG.nii.gz")
   #voxel_vol = nii2numpy(nii_path)
    false_neg_vol,false_pos_vol,dice_sc =0,0,0
    h,w,d = config.target_size
    volume_size = h*w*d
    for i in range(batch_size):
        # false_neg_vol += false_neg_pix(gt_array[i], pred_array[i])/volume_size# * voxel_vol
        # false_pos_vol += false_pos_pix(gt_array[i], pred_array[i])/volume_size #* voxel_vol
        dice_sc += dice_score(gt_array[i], pred_array[i])

    return dice_sc/batch_size, false_pos_vol/batch_size, false_neg_vol/batch_size

if __name__ == '__main__':
    from TrainConfig import  config
    ones = np.ones((1,20,20,20))
    ones = np.ones((1,20,20,20))
    zeros = np.zeros((1,20,20,20))
    print(compute_metrics(ones,ones,config))
    print(compute_metrics(ones,zeros,config))

