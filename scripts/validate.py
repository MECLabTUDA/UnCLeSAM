import sys
sys.path.append('/gris/gris-f/homelv/maflal/SAM')

import numpy as np
from model.utils import *
import itertools, monai, sys
import torch, time, os, argparse
from model.sam_networks import Own_Sam
from model.sam_without_adapter import Sam_Without_Adapter
from decorators.loggers import log_prints
from dataloading.generators import npz_generator

import wandb

def validate_no_adapter(model, val_list, epoch, out_, store_samples=False, npz_=True):
    r"""
    Call this function in order to validate the model with data from the provided generator.
        :param model: The model architecture to use for predictions.
        :param val_list: List of paths to pre-processed validation cases (.npz).
        :param epoch: Integer of the current epoch we perform the validation in
        :param out_: Path to where the metrics (and samples) should be stored
        :param store_samples: Set this flag if png images should be generated showing the samples and bounding boxes
        :param npz_: Indicates if the list contains paths to npz files or not. In the latter the pre-processing will be performed.
    """
    # Put model in eval mode and initialize dictionaries
    model.eval()
    # val_res = pd.DataFrame()
    device = next(model.parameters()).device

    for npz in val_list:
        # Extract the case name
        case = npz.split(os.sep)[-1].split('.npz')[0]
        task = npz.split(os.sep)[-4]
        # Load input image (in this case we already pre-processed it using one of our scripts)
        if npz_:
            imgs_, segs_, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt = d_utils.load_npzfile(npz, model.nr_samples)
        else:
            # TODO: Add alternative if the image is nii.gz and not a npz file, i.e. not pre-processed
            pass
        
        imgs = torch.from_numpy(imgs_).to(device).float()
        segs = torch.from_numpy(segs_).to(device).float()
        embeds = torch.from_numpy(embeds).to(device).float()
        bbox = torch.from_numpy(bboxs_gt).to(device).float()    # (n, 4)
        grid = torch.from_numpy(samples_gt).to(device).float()    # (n, nr_samples, 2)

        # grid = get_grid(256, 10).unsqueeze(0).repeat(imgs.shape[0],1 ,1)
        # bbox = torch.randn(4).unsqueeze(0).repeat(imgs.shape[0],1,1) # just for now
        
        # Predict
        _y_pred_, _, _, samples, bbox = model(imgs, embeds, orig_sizes, input_sizes, bbox, grid, train=False)
        # _y_pred_, _, _, samples, bbox= model(imgs, embeds, orig_sizes, input_sizes, train=False)    # Only use the ones for plotting here

        # Prepare predictions and GT
        nb_classes = segs.size(1) + 1 # Don't forget the background
        y_pred_ = get_one_hot(_y_pred_, nb_classes)
        segs = get_one_hot(segs.detach().cpu().numpy().squeeze().astype(np.uint8), nb_classes)

        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HD = HausdorffDistanceMetric(include_background=False)(y_pred_, segs)
        Dice = DiceMetric(include_background=False, ignore_empty=False)(y_pred_, segs)
        IoU = MeanIoU(include_background=False, ignore_empty=False)(y_pred_, segs)
        MSE_bbox = MSEMetric()(bbox, torch.from_numpy(bboxs_gt))
        MSE_samples = MSEMetric()(samples, torch.from_numpy(samples_gt))
        
        val_res = {'Epoch': str(epoch), 'Task': task, 'ID': case, 'HD': np.mean(HD.numpy()),
                    'Dice': np.mean(Dice.numpy()*100),
                    'IoU': np.mean(IoU.numpy()*100),
                    'MSE (samples)': np.mean(MSE_samples.numpy()),
                    'MSE (bbox)': np.mean(MSE_bbox.numpy()),
                    }


        # Store pngs with samples to show progress during training if desired
        if store_samples:
            os.makedirs(out_, exist_ok=True)
            # Store a png with bounding box and samples for every slice
            out = os.path.join(out_, "Epoch_"+str(epoch), task, case)
            os.makedirs(out, exist_ok=True)
            os.makedirs(os.path.join(out, "GT"), exist_ok=True)
            
            # Generate for every slice the pngs and store them under out/case_slice_ID.png
            for i, slice in enumerate(imgs_):
                # Plot the created GT labels and bounding box use this line below
                plot_slice_with_samples_bbox_png(slice, samples_gt[i],
                                                 bboxs_gt[i], os.path.join(out, "GT", "slice_"+str(i)+'.png'))
                # Plot the predicted samples and bbox as well
                plot_slice_with_samples_bbox_png(slice, samples[i].detach().cpu().numpy(),
                                                 bbox[i].detach().cpu().numpy(), os.path.join(out, "slice_"+str(i)+'.png'))

    # Put model back into train mode and return the results
    model.train()
    return y_pred_, samples, bbox, val_res, _y_pred_  # --> patient wise, not batch wise

def validate_all(model, adapter, val_list, epoch, out_, store_samples=False, npz_=True):
    r"""
    Call this function in order to validate the model with data from the provided generator.
        :param model: The model architecture to use for predictions.
        :param val_list: List of paths to pre-processed validation cases (.npz).
        :param epoch: Integer of the current epoch we perform the validation in (if None folder with epoch won't be added to output folder)
        :param out_: Path to where the metrics (and samples) should be stored
        :param store_samples: Set this flag if png images should be generated showing the samples and bounding boxes
        :param npz_: Indicates if the list contains paths to npz files or not. In the latter the pre-processing will be performed.
    """
    # Put model in eval mode and initialize dictionaries
    model.eval()
    adapter.eval()
    val_res = pd.DataFrame()
    device = next(model.parameters()).device

    for npz in val_list:
        # Extract the case name
        case = npz.split(os.sep)[-1].split('.npz')[0]
        task = npz.split(os.sep)[-4]
        # Load input image (in this case we already pre-processed it using one of our scripts)
        if npz_:
            imgs_, segs_, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt = d_utils.load_npzfile(npz, model.nr_samples)
        else:
            # TODO: Add alternative if the image is nii.gz and not a npz file, i.e. not pre-processed
            pass
        
        imgs = torch.from_numpy(imgs_).to(device).float()
        segs = torch.from_numpy(segs_).to(device).float()
        embeds = torch.from_numpy(embeds).to(device).float()

        samples_adapt, bbox_adapt = adapter(embeds)
        # imgs = imgs.permute(0, 3, 1, 2)
        # samples_adapt, bbox_adapt = adapter(imgs)
        # bbox_adapt = torch.from_numpy(bboxs_gt).to(device).float()    # (n, 4)
        # samples_adapt = torch.from_numpy(samples_gt).to(device).float()    # (n, nr_samples, 2)

        # grid = get_grid(256, 10).unsqueeze(0).repeat(imgs.shape[0],1 ,1)
        # bbox = torch.randn(4).unsqueeze(0).repeat(imgs.shape[0],1,1) # just for now
        
        # Predict
        _y_pred_, _, _, samples, bbox = model(imgs, embeds, orig_sizes, input_sizes, bbox_adapt, samples_adapt, train=False)
        # _y_pred_, _, _, samples, bbox= model(imgs, embeds, orig_sizes, input_sizes, train=False)    # Only use the ones for plotting here

        # Prepare predictions and GT
        nb_classes = segs.size(1) + 1 # Don't forget the background
        y_pred_ = get_one_hot(_y_pred_, nb_classes)
        segs = get_one_hot(segs.detach().cpu().numpy().squeeze().astype(np.uint8), nb_classes)

        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HD = HausdorffDistanceMetric(include_background=False)(y_pred_, segs)
        Dice = DiceMetric(include_background=False, ignore_empty=False)(y_pred_, segs)
        IoU = MeanIoU(include_background=False, ignore_empty=False)(y_pred_, segs)
        MSE_bbox = MSEMetric()(bbox, torch.from_numpy(bboxs_gt))
        MSE_samples = MSEMetric()(samples, torch.from_numpy(samples_gt))

        # Append to dataframe
        val_res = pd.concat([val_res,
                                  pd.DataFrame.from_records([{'Epoch': str(epoch), 'Task': task, 'ID': case, 'HD': np.mean(HD.numpy()),
                                                              'Dice': np.mean(Dice.numpy()*100),
                                                              'IoU': np.mean(IoU.numpy()*100),
                                                              'MSE (samples)': np.mean(MSE_samples.numpy()),
                                                              'MSE (bbox)': np.mean(MSE_bbox.numpy()),
                                                              }])
                            ], axis=0)
        
        # Store pngs with samples to show progress during training if desired
        if store_samples:
            os.makedirs(out_, exist_ok=True)
            # Store a png with bounding box and samples for every slice
            if epoch is None:
                out = os.path.join(out_, task, case)
            else:
                out = os.path.join(out_, "Epoch_"+str(epoch), task, case)
            os.makedirs(out, exist_ok=True)
            os.makedirs(os.path.join(out, "GT"), exist_ok=True)
            
            # Generate for every slice the pngs and store them under out/case_slice_ID.png
            for i, slice in enumerate(imgs_):
                # Plot the created GT labels and bounding box use this line below
                plot_slice_with_samples_bbox_png(slice, samples_gt[i],
                                                 bboxs_gt[i], os.path.join(out, "GT", "slice_"+str(i)+'.png'), seg=segs_[i][0])
                # Plot the predicted samples and bbox as well
                plot_slice_with_samples_bbox_png(slice, samples[i].detach().cpu().numpy(),
                                                 bbox[i].detach().cpu().numpy(), os.path.join(out, "slice_"+str(i)+'.png'),
                                                 seg=y_pred_.argmax(1).detach().cpu().numpy().squeeze().astype(np.uint8)[i])

    # Store metrics csv
    if not os.path.isfile(os.path.join(out_, "validation_results.csv")):
        val_res.to_csv(os.path.join(out_, "validation_results.csv"), index=False, sep=',')  # <-- Includes the header
    else: # else it exists so append without writing the header
        val_res.to_csv(os.path.join(out_, "validation_results.csv"), index=False, sep=',', mode='a', header=False)  # <-- Omits the header 

    # Put model back into train mode and return the results
    model.train()
    adapter.train()
    return y_pred_, samples, bbox, val_res, _y_pred_  # --> patient wise, not batch wise

def validate_adapter(model, adapter, val_list, epoch, out_, store_samples=False, npz_=True):
    r"""
    Call this function in order to validate the model with data from the provided generator.
        :param model: The model architecture to use for predictions.
        :param val_list: List of paths to pre-processed validation cases (.npz).
        :param epoch: Integer of the current epoch we perform the validation in
        :param out_: Path to where the metrics (and samples) should be stored
        :param store_samples: Set this flag if png images should be generated showing the samples and bounding boxes
        :param npz_: Indicates if the list contains paths to npz files or not. In the latter the pre-processing will be performed.
    """
    # Put model in eval mode and initialize dictionaries
    model.eval()
    adapter.eval()
    val_res = pd.DataFrame()
    device = next(model.parameters()).device

    for npz in val_list:
        # Extract the case name
        case = npz.split(os.sep)[-1].split('.npz')[0]
        task = npz.split(os.sep)[-4]
        # Load input image (in this case we already pre-processed it using one of our scripts)
        if npz_:
            imgs_, segs_, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt = d_utils.load_npzfile(npz, model.nr_samples)
        else:
            # TODO: Add alternative if the image is nii.gz and not a npz file, i.e. not pre-processed
            pass
        
        imgs = torch.from_numpy(imgs_).to(device).float()
        segs = torch.from_numpy(segs_).to(device).float()
        embeds = torch.from_numpy(embeds).to(device).float()

        samples_adapt, bbox_adapt = adapter(embeds)
        # imgs = imgs.permute(0, 3, 1, 2)
        # samples_adapt, bbox_adapt = adapter(imgs)
        # bbox_adapt = torch.from_numpy(bboxs_gt).to(device).float()    # (n, 4)
        # samples_adapt = torch.from_numpy(samples_gt).to(device).float()    # (n, nr_samples, 2)

        # grid = get_grid(256, 10).unsqueeze(0).repeat(imgs.shape[0],1 ,1)
        # bbox = torch.randn(4).unsqueeze(0).repeat(imgs.shape[0],1,1) # just for now
        
        # Predict
        _y_pred_, _, _, samples, bbox = model(imgs, embeds, orig_sizes, input_sizes, bbox_adapt, samples_adapt, train=False)
        # _y_pred_, _, _, samples, bbox= model(imgs, embeds, orig_sizes, input_sizes, train=False)    # Only use the ones for plotting here

        # Prepare predictions and GT
        nb_classes = segs.size(1) + 1 # Don't forget the background
        y_pred_ = get_one_hot(_y_pred_, nb_classes)
        segs = get_one_hot(segs.detach().cpu().numpy().squeeze().astype(np.uint8), nb_classes)

        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HD = HausdorffDistanceMetric(include_background=False)(y_pred_, segs)
        Dice = DiceMetric(include_background=False, ignore_empty=False)(y_pred_, segs)
        IoU = MeanIoU(include_background=False, ignore_empty=False)(y_pred_, segs)
        MSE_bbox = MSEMetric()(bbox, torch.from_numpy(bboxs_gt))
        MSE_samples = MSEMetric()(samples, torch.from_numpy(samples_gt))
        
        # val_res = {'Epoch': str(epoch), 'Task': task, 'ID': case, 'HD': np.mean(HD.numpy()),
        #             'Dice': np.mean(Dice.numpy()*100),
        #             'IoU': np.mean(IoU.numpy()*100),
        #             'MSE (samples)': np.mean(MSE_samples.numpy()),
        #             'MSE (bbox)': np.mean(MSE_bbox.numpy()),
        #             }


        val_res = pd.concat([val_res,
                                  pd.DataFrame.from_records([{'Epoch': str(epoch), 'Task': task, 'ID': case, 'HD': np.mean(HD.numpy()),
                                                              'Dice': np.mean(Dice.numpy()*100),
                                                              'IoU': np.mean(IoU.numpy()*100),
                                                              'MSE (samples)': np.mean(MSE_samples.numpy()),
                                                              'MSE (bbox)': np.mean(MSE_bbox.numpy()),
                                                              }])
                            ], axis=0)

        # Store pngs with samples to show progress during training if desired
        if store_samples:
            os.makedirs(out_, exist_ok=True)
            # Store a png with bounding box and samples for every slice
            out = os.path.join(out_, "Epoch_"+str(epoch), task, case)
            os.makedirs(out, exist_ok=True)
            os.makedirs(os.path.join(out, "GT"), exist_ok=True)
            
            # Generate for every slice the pngs and store them under out/case_slice_ID.png
            for i, slice in enumerate(imgs_):
                # Plot the created GT labels and bounding box use this line below
                plot_slice_with_samples_bbox_png(slice, samples_gt[i],
                                                 bboxs_gt[i], os.path.join(out, "GT", "slice_"+str(i)+'.png'))
                # Plot the predicted samples and bbox as well
                plot_slice_with_samples_bbox_png(slice, samples[i].detach().cpu().numpy(),
                                                 bbox[i].detach().cpu().numpy(), os.path.join(out, "slice_"+str(i)+'.png'))

    # Put model back into train mode and return the results
    model.train()
    adapter.train()
    return y_pred_, samples, bbox, val_res, _y_pred_  # --> patient wise, not batch wise