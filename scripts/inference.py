#!/usr/bin/env python

"""
Example script to segment using our code base
"""

import sys
sys.path.append('/gris/gris-f/homelv/maflal/SAM')

import SimpleITK as sitk
from model.utils import *
import os, argparse, shutil, sys
from model.sam_without_adapter import *
import dataloading.utils as d_utils
from validate import validate_no_adapter

import matplotlib.pyplot as plt

def segment():
    args = get_args()
    return _segment(args)

def get_args():
    # parse commandline args
    parser = argparse.ArgumentParser()
    # parser.add_argument('--out', required=True, help='out path')
    # parser.add_argument('--gt', required=True, help='GT segmentation path')
    parser.add_argument('--npz', required=True, help='pre-processed img npz file with embeddings and other stuff')
    parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--store_npz', action='store_true', help='Set this if the input should be copied as well --> takes a lot of space..')
    
    args = parser.parse_args()
    return args

# @log_prints(LOG_FILE)
def _segment(args):
    # device handling
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load and set up model
    model = Sam_Without_Adapter.load(args.model, device)
    model.to(device)
    model.eval()

    # load input image (in this case we already pre-processed it using one of our scripts)
    imgs, segs, _, _, _, _, _ = d_utils.load_npzfile(args.npz, model.nr_samples)

    # Extract meta information on model
    # epoch = int(args.model.split(os.sep)[-1][:-3])
    # model_dir = os.path.join(os.sep, *args.model.split(os.sep)[:-1])
    
    # Do validation for one sample based on args.npz
    _, _, _, val_res, y_pred_ = validate_no_adapter(model, [args.npz], 0, out_="pretrained/inf")

    # # Store npz finput file
    # if args.store_npz:
    #     shutil.copy(args.npz, os.path.join(args.out, 'input.npz'))

    # # Store image and GT segmentation
    # sitk.WriteImage(sitk.GetImageFromArray(imgs[..., 0]), os.path.join(args.out, 'img.nii.gz'))
    # sitk.WriteImage(sitk.GetImageFromArray(segs.squeeze()), os.path.join(args.out, 'seg_gt.nii.gz'))

    # # Store predicted segmentation
    # sitk.WriteImage(sitk.GetImageFromArray(y_pred_), os.path.join(args.out, 'pred_seg.nii.gz'))

    # Randomly select one image from imgs
    ind = 10  # Change this to select a different image randomly
    selected_img = imgs[ind]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ground truth segmentation mask
    axs[0].imshow(selected_img, cmap='gray')
    axs[0].imshow(segs.squeeze()[ind], alpha=0.5, cmap='jet')
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    # Plot predicted segmentation mask
    axs[1].imshow(selected_img, cmap='gray')
    axs[1].imshow(y_pred_[ind], alpha=0.5, cmap='jet')
    axs[1].set_title('Prediction')
    axs[1].axis('off')

    # save the image
    fig.savefig('pretrained/inf/seg.png')

    # return val_res.loc[:, 'HD'].mean(), val_res.loc[:, 'Dice'].mean(), val_res.loc[:, 'IoU'].mean()

# -- Main function for setup execution -- #
def main():
    segment()

if __name__ == "__main__":
    segment()