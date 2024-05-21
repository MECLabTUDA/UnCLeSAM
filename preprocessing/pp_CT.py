#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 6 19:25:36 2023

convert CT nii image to npz files, including input image, image embeddings, and ground truth

@author: jma

Extracted from: https://github.com/bowang-lab/MedSAM/blob/main/pre_CT.py and modified for our needs.
"""

# import packages
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import os, argparse, torch
from model.utils import set_all_seeds
from skimage import transform, io, segmentation
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
join = os.path.join 

# Set seeds for numpy, random and pytorch
set_all_seeds(3299)

task = "Task097_DecathHip"
model_type = 'vit_b'
checkpoint = '/home/aranem_locale/Desktop/SAM_CL/checkpoints/sam_vit_b_01ec64.pth'
device = 'cuda:5'
# # save testing data
# for name in tqdm(test_names):
#     image_name = name.split('.nii.gz')[0] + args.img_name_suffix
#     gt_name = name 
#     imgs, gts = preprocess_ct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model=None, device=args.device)
#     # save to npz file
#     if len(imgs)>1:
#         imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
#         gts = np.stack(gts, axis=0) # (n, 256, 256)
#         img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
#         np.savez_compressed(join(save_path_ts, prefix + '_' + gt_name.split('.nii.gz')[0]+'.npz'), imgs=imgs, gts=gts)
#         # save an example image for sanity check
#         idx = np.random.randint(0, imgs.shape[0])
#         img_idx = imgs[idx,:,:,:]
#         gt_idx = gts[idx,:,:]
#         bd = segmentation.find_boundaries(gt_idx, mode='inner')
#         img_idx[bd, :] = [255, 0, 0]
#         io.imsave(save_path_ts + '.png', img_idx, check_contrast=False)

# set up the parser
parser = argparse.ArgumentParser(description='preprocess CT images')
parser.add_argument('-i', '--nii_path', type=str, default=f'/home/aranem_locale/Desktop/mnts/local/scratch/aranem/WACV_2024_SAM/WACV_2024_raw_data/{task}/imagesTr', help='path to the nii images')
parser.add_argument('-gt', '--gt_path', type=str, default=f'/home/aranem_locale/Desktop/mnts/local/scratch/aranem/WACV_2024_SAM/WACV_2024_raw_data/{task}/labelsTr', help='path to the ground truth',)
parser.add_argument('-o', '--npz_path', type=str, default=f'/home/aranem_locale/Desktop/mnts/local/scratch/aranem/WACV_2024_SAM/WACV_2024_preprocessed_data/{task}/{model_type}/Tr', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--modality', type=str, default='MRI', help='modality')
# parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='anatomy')
parser.add_argument('--img_name_suffix', type=str, default='_0000.nii.gz', help='image name suffix')
parser.add_argument('--label_id', type=int, default=1, help='label id')
# parser.add_argument('--prefix', type=str, default='CT_Abd-Gallbladder_', help='prefix')
parser.add_argument('--model_type', type=str, default=model_type, help='model type')
parser.add_argument('--checkpoint', type=str, default=checkpoint, help='checkpoint')
parser.add_argument('--device', type=str, default=device, help='device')
args = parser.parse_args()

# prefix = args.modality + '_' + args.anatomy
names = sorted(os.listdir(args.gt_path))
names = [name for name in names if not os.path.exists(join(args.npz_path, name.split('.nii.gz')[0]+'.npz'))]

# def preprocessing function
def preprocess_ct(gt_path, nii_path, gt_name, image_name, label_id, image_size, sam_model, device):
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data = np.uint8(gt_data==label_id)

    imgs = []
    gts =  []
    img_embeddings = []

    assert (np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2) or (np.max(gt_data)==0 and np.unique(gt_data).shape[0]==1), 'ground truth should be binary or empty'

    img_sitk = sitk.ReadImage(join(nii_path, image_name))
    image_data_ = sitk.GetArrayFromImage(img_sitk)
    # nii preprocess start
    lower_bound = -500
    upper_bound = 1000
    image_data_pre = np.clip(image_data_, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
    image_data_pre[image_data_==0] = 0
    image_data_pre = np.uint8(image_data_pre)
    
    z_index, _, _ = np.where(gt_data>0) if np.sum(gt_data)>0 else np.where(gt_data==0)  # Also allow empty GTs
    z_min, z_max = np.min(z_index), np.max(z_index)
    
    for i in range(z_min, z_max):
        gt_slice_i = gt_data[i,:,:]
        gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        if np.sum(gt_slice_i)>100:
            # resize img_slice_i to 256x256
            img_slice_i = transform.resize(image_data_pre[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            # convert to three channels
            img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
            assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
            assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
            imgs.append(img_slice_i)
            # assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
            gts.append(gt_slice_i)
            if sam_model is not None:
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(img_slice_i)
                # resized_shapes.append(resize_img.shape[:2])
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                # model input: (1, 3, 1024, 1024)
                input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                
                # input_imgs.append(input_image.cpu().numpy()[0])
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])

    if sam_model is not None:
        return image_data_, imgs, gts, img_embeddings, original_size, input_size
    else:
        return image_data_, imgs, gts

os.makedirs(args.npz_path, exist_ok=True)

# set up the model
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(args.device)

for name in tqdm(names):
    image_name = name.split('.nii.gz')[0] + args.img_name_suffix
    gt_name = name
    orig, imgs, gts, img_embeddings, original_size, input_size = preprocess_ct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model, args.device)
    # save to npz file
    # stack the list to array
    # if len(imgs)>1:
    # orig (n, h, w)
    imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
    gts = np.stack(gts, axis=0) # (n, 256, 256)
    img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64) --> (n, 256, 64, 64)
    np.savez_compressed(join(args.npz_path, gt_name.split('.nii.gz')[0]+'.npz'), img_orig=orig, imgs=imgs, gts=gts,
                        img_embeddings=img_embeddings, original_size=original_size, input_size=input_size)
    # save an example image for sanity check
    idx = np.random.randint(0, imgs.shape[0])
    img_idx = imgs[idx,:,:,:]
    gt_idx = gts[idx,:,:]
    bd = segmentation.find_boundaries(gt_idx, mode='inner')
    img_idx[bd, :] = [255, 0, 0]
    io.imsave(join(args.npz_path, name.split('.nii.gz')[0]+'.png'), img_idx, check_contrast=False)
    sitk.WriteImage(sitk.GetImageFromArray(imgs), join(args.npz_path, name.split('.nii.gz')[0]+'.nii.gz')) 
    sitk.WriteImage(sitk.GetImageFromArray(gts), join(args.npz_path, name.split('.nii.gz')[0]+'_gt.nii.gz')) 