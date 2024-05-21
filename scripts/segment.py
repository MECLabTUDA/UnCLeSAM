#!/usr/bin/env python

"""
Example script to segment using our code base
"""

import SimpleITK as sitk
from validate import validate_all
import os, argparse, shutil, sys
from model.sam_networks import *
import dataloading.utils as d_utils
from decorators.loggers import log_prints
from model.sam_without_adapter import Sam_Without_Adapter
from model.sam_adapters import ConvAdapter

# Extract the LOG_FILE --> If not set, the decorator won't do anything
# Make sure that this file  is important AFTER setting sys.argv, otheriwse LOG_FILE will be None
try:
    LOG_FILE = sys.argv[sys.argv.index('--log')+1]
except Exception as e:
    print(e)
    LOG_FILE = None

def segment():
    args = get_args()
    return _segment(args)

def get_args():
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='out path')
    # parser.add_argument('--gt', required=True, help='GT segmentation path')
    parser.add_argument('--npz', required=True, help='pre-processed img npz file with embeddings and other stuff')
    parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--store_npz', action='store_true', help='Set this if the input should be copied as well --> takes a lot of space..')
    parser.add_argument('--log', type=str, default=None, help='Set this and provide a log file path if the prints hould be logged.')
    
    args = parser.parse_args()
    return args

# @log_prints(LOG_FILE)
def _segment(args):
    # device handling
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # load and set up model
    model = Sam_Without_Adapter.load(args.model, device)
    adapter = ConvAdapter.load(args.model.replace('best_model.pt', 'best_adapter.pt'), device)
    model.to(device)
    model.eval()
    adapter.to(device)
    adapter.eval()

    # load input image (in this case we already pre-processed it using one of our scripts)
    imgs, segs, _, _, _, _, _ = d_utils.load_npzfile(args.npz, model.nr_samples)

    # Extract meta information on model
    # epoch = int(args.model.split(os.sep)[-1][:-9].replace('_model', ''))
    model_dir = os.path.join(os.sep, *args.model.split(os.sep)[:-1])
    
    # Do validation for one sample based on args.npz
    _, _, _, val_res, y_pred_ = validate_all(model, adapter, [args.npz], epoch=None, store_samples=True, out_=os.path.join(model_dir, "inference"))

    # Store npz finput file
    if args.store_npz:
        shutil.copy(args.npz, os.path.join(args.out, 'input.npz'))

    # Store image and GT segmentation
    sitk.WriteImage(sitk.GetImageFromArray(imgs[..., 0]), os.path.join(args.out, 'img.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(segs.squeeze()), os.path.join(args.out, 'seg_gt.nii.gz'))

    # Store predicted segmentation
    sitk.WriteImage(sitk.GetImageFromArray(y_pred_), os.path.join(args.out, 'pred_seg.nii.gz'))

    return val_res.loc[:, 'HD'].mean(), val_res.loc[:, 'Dice'].mean(), val_res.loc[:, 'IoU'].mean()

# -- Main function for setup execution -- #
def main():
    segment()

if __name__ == "__main__":
    segment()