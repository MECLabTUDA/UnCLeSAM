#!/usr/bin/env python

"""
Example script to train a SAM model with Adapter.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

EPSILON = 1e-8
ALPHA = 0.9
LAMBDA = 0.4

import pickle

import sys
# sys.path.append('/gris/gris-f/homelv/maflal/SAM')

import numpy as np
from model.utils import *
import itertools, monai, sys
from model.losses import RWalkLoss
import torch, time, os, argparse
import torch.nn as nn
from model.sam_networks import Own_Sam
from model.sam_without_adapter import Sam_Without_Adapter
from model.sam_adapters import ConvAdapter
from decorators.loggers import log_prints
from dataloading.generators import npz_generator
from validate import validate_adapter

import wandb

# Set seeds for numpy, random and pytorch
set_all_seeds(3299)
torch.set_printoptions(profile="full")

# Extract the LOG_FILE --> If not set, the decorator won't do anything
# Make sure that this file  is important AFTER setting sys.argv, otheriwse LOG_FILE will be None
try:
    LOG_FILE = sys.argv[sys.argv.index('--log')+1]
except Exception as e:
    print(e)
    LOG_FILE = None

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def train():
    args = get_args()
    _train(args)

def get_args():
    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--train-dir', required=True, help='path to folder with pre-processed train files (.npz)')
    parser.add_argument('--val-dir', required=True, help='path to folder with pre-processed val files (.npz)')
    parser.add_argument('--model-dir', default='',
                        help='model output directory.')
    parser.add_argument('--freeze_sam', action='store_true',
                        help='Set this if SAm should be frozen completely.')
    parser.add_argument('--task', required=True, help='Task Name we train on')
    parser.add_argument('--samples',  type=int, default=100,
                        help='Specify the amount of samples that should be extracted.')
    parser.add_argument('--save_steps',  type=int, default=25,
                        help='Specify after how many epochs the model state is saved and validation is performed.')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of training epochs (default: 250)')
    parser.add_argument('--steps-per-epoch', type=int, default=250,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')
    parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/sam_vit_b.pth', help='checkpoint')
    parser.add_argument('--log', type=str, default=None, help='Set this and provide a log file path if the prints should be logged.')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',help='Name of the experiment')

    # loss hyperparameters
    # parser.add_argument('--image-loss', default='ncc',
    #                     help='image reconstruction loss - can be mse or ncc (default: mse)')
    # parser.add_argument('--lambda', type=float, dest='weight', default=1,
    #                     help='weight of deformation loss (default: 0.01)')
    args = parser.parse_args()
    return args

# @log_prints(LOG_FILE)
def _train(args):

    wandb.init(project="SAM-CL", name=args.exp_name)

    freeze_all = args.freeze_sam

    # Extract vol names
    train_files = [os.path.join(args.train_dir, x) for x in os.listdir(args.train_dir) if '.npz' in x]
    # val_files = [os.path.join(args.val_dir, x) for x in os.listdir(args.val_dir) if '.npz' in x]

    # -- Group for tasks to get eval splits between tasks -- #
    train_lists = [list(v) for _, v in itertools.groupby(train_files, key=lambda x:x.split(os.sep)[-3])]  # --> task names and IDs

    for x in train_lists:
        random.shuffle(x)
    
    # -- Split into train and val based on tasks using 80:20 split -- #
    train_files_train = [x[:int((len(x)+1)*.80)] for x in train_lists] # Remaining 80% to training set
    train_files_val = [x[int((len(x)+1)*.80):] for x in train_lists] # Splits 20% data to test set

    # -- Join the list of lists -- #
    train_files, val_files = [item for sublist in train_files_train for item in sublist], [item for sublist in train_files_val for item in sublist]

    # load and prepare training data
    generator = npz_generator(train_files, nr_samples=args.samples, batch_size=args.batch_size)

    # extract shape from sampled input
    inshape = next(generator)[0][0].shape   # (256, 256, 3)
    
    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # assert np.mod(args.batch_size, nb_gpus) == 0, \
    #     'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
    
    if args.load_model:
        # load initial model (if specified)
        model = Sam_Without_Adapter.load(args.load_model, device)
        adapter = ConvAdapter.load(args.load_model.replace('_model.pt', '_adapter.pt'), device)
        fisher = load_pickle(os.path.join(os.sep, *args.load_model.split(os.sep)[:-1], 'fisher.pkl'))
        params = load_pickle(os.path.join(os.sep, *args.load_model.split(os.sep)[:-1], 'params.pkl'))
        scores = load_pickle(os.path.join(os.sep, *args.load_model.split(os.sep)[:-1], 'scores.pkl'))
    else:
        fisher, params, scores = dict(), dict(), dict()
        # otherwise configure new model
        model = Sam_Without_Adapter(inshape=inshape, nr_samples=args.samples, model_type=args.model_type, checkpoint=args.checkpoint, device=device)
        # Define Adapter
        adapter = ConvAdapter(nr_samples=args.samples)
    prev_param = None

    # -- Define the fisher and params before the training -- #
    if not freeze_all:
        fisher[args.task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False) for m in [adapter, model] for n, p in m.named_parameters() if p.requires_grad}
        params[args.task] = dict()
        scores[args.task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False) for m in [adapter, model] for n, p in m.named_parameters() if p.requires_grad}
    else:
        fisher[args.task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False) for m in [adapter] for n, p in m.named_parameters() if p.requires_grad}
        params[args.task] = dict()
        scores[args.task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False) for m in [adapter] for n, p in m.named_parameters() if p.requires_grad}

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()
    adapter.to(device)
    adapter.train()

    wandb.watch(model)

    # set optimizer
    # optimizer = torch.optim.Adam(list(model.parameters()) + list(adapter.parameters()), lr=args.lr, weight_decay=1e-4)
    if not freeze_all:  # Only when not frozen
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_adapter = torch.optim.Adam(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

    # prepare loss with Dice
    seg_criterion = monai.losses.DiceCELoss(sigmoid=True, reduction='mean')
    prompt_criterion = nn.MSELoss(reduction='mean').to(device)

    weights = [1, 1, 1]


    if len([x for x in fisher.keys() if x != args.task]) > 0:
        rwalk_citerion = RWalkLoss(fisher=fisher, params=params, parameter_importance=scores, ewc_lambda=LAMBDA)
        weights.append(1)

    # Define variables for early stopping
    best_val_dice = 0.0
    patience = 15  # Number of epochs to wait for improvement
    counter = 0  # Counter to track epochs without improvement

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):
        # save model checkpoint
        if epoch % args.save_steps == 0:
            model.save(os.path.join(model_dir, '%04d_model.pt' % epoch))
            model.save(os.path.join(model_dir, '%04d_adapter.pt' % epoch))

            # -- Update the importance score using distance in Riemannian Manifold -- #
            if prev_param is not None:
                for m in [adapter, model] if not freeze_all else [adapter]:
                    for name, param in m.named_parameters():
                        if param.grad is not None:
                            # -- Get parameter difference from old param and current param t -- #
                            delta = param.grad.detach() * (prev_param[name].to(param.device) - param.detach())
                            delta = delta.to(0)
                            # -- Calculate score denominator -- #
                            den = 0.5 * fisher[args.task][name] * (param.detach() - prev_param[name].to(param.device)).pow(2).to(0) + EPSILON
                            # -- Score: delat(L) / 0.5*F_t*delta(param)^2 --> only positive or zero values -- #
                            scores_ = (delta / den)
                            scores_[scores_ < 0] = 0  # Ensure no negative values
                            # -- Update the scores -- #
                            scores[args.task][name] += scores_

            # -- Update the prev params -- #
            if epoch != 0:
                if not freeze_all:
                    prev_param = {k: torch.clone(v).detach().cpu() for m in [model, adapter] for k, v in m.named_parameters() if v.grad is not None}
                else:
                    prev_param = {k: torch.clone(v).detach().cpu() for m in [adapter] for k, v in m.named_parameters() if v.grad is not None}

            # -- Update the fisher values -- #
            
            for m in [model, adapter] if not freeze_all else [adapter]:
                for name, param in m.named_parameters():
                    # -- F_t = alpha * F_t + (1-alpha) * F_t-1
                    if param.grad is not None:
                        f_t = param.grad.data.clone().pow(2).to(0)
                        f_to = fisher[args.task][name] if fisher[args.task][name] is not None else torch.tensor([0], device='cuda:0')
                        fisher[args.task][name] = (ALPHA * f_t) + ((1 - ALPHA) * f_to)

                for name, param in m.named_parameters():
                    # -- Update the params dict -- #
                    params[args.task][name] = param.data.clone()

            write_pickle(fisher, os.path.join(model_dir, 'fisher.pkl'))
            write_pickle(params, os.path.join(model_dir, 'params.pkl'))
            write_pickle(scores, os.path.join(model_dir, 'scores.pkl'))

        epoch_loss = []
        epoch_seg_loss = []
        epoch_bbox_loss = []
        epoch_rwalk_loss = []
        epoch_samples_loss = []
        epoch_step_time = []
        dice = []
        wandb_log = {}

        for _ in range(args.steps_per_epoch):

            step_start_time = time.time()

            # Generate inputs (and true outputs) and convert them to tensors
            imgs, segs, embeds, orig_sizes, input_sizes, bboxs_gt_, samples_gt_ = next(generator)
            imgs = torch.from_numpy(imgs).to(device).float()    # (n, 256, 256, 3)
            segs = torch.from_numpy(segs).to(device).float()    # (n, 1, 256, 256)
            embeds = torch.from_numpy(embeds).to(device).float()    # (n, 1, 256, 64, 64)
            bboxs_gt = torch.from_numpy(bboxs_gt_).to(device).float()    # (n, 4)
            samples_gt = torch.from_numpy(samples_gt_).to(device).float()    # (n, nr_samples, 2)

            loss = 0
            loss_list = []

            samples_adapt, bbox_adapt = adapter(embeds)
            # imgs = imgs.permute(0, 3, 1, 2)
            # samples_adapt, bbox_adapt = adapter(imgs)

            # Run inputs through the model
            y_pred_, samples_coords_for_loss, bbox_coords_for_loss, _, _ = model(imgs, embeds, orig_sizes, input_sizes, bbox_adapt, samples_adapt, train=True, freeze_all=freeze_all)

            # Flatten the GT samples from [N, nr_samples, 2] --> [n, 2*nr_samples]
            samples_gt_flat = torch.flatten(samples_gt, start_dim=1)    # Size([n, 2*nr_samples])
            
            # Perform bboxs into binary mask to calculate the Dice as well
            seg_mask_size = tuple(segs.squeeze().size())
            nb_classes = segs.size(1) + 1 # Don't forget the background)
            bbox_mask_gt = torch.from_numpy(get_mask_from_bbox_coord(bboxs_gt, seg_mask_size)).to(device).unsqueeze(1) # Size([N, 1, 256, 256])
            
            # Build true and pred lists for loss calculation with Dice
            y_pred = y_pred_
            y_true = segs.long()
            
            # Calculate total loss
            seg_loss = seg_criterion(y_pred, y_true)
            adapt_loss_bbox = prompt_criterion(bbox_adapt, bboxs_gt)
            adapt_loss_samples = prompt_criterion(samples_adapt, samples_gt)

            # loss = seg_criterion(y_pred, y_true)
            if len([x for x in fisher.keys() if x != args.task]) > 0:
                rwalk_loss = rwalk_citerion.loss(adapter.named_parameters())
                if not freeze_all:
                    rwalk_loss += rwalk_citerion.loss(model.named_parameters())
                loss = seg_loss + adapt_loss_bbox + adapt_loss_samples + rwalk_loss
                epoch_rwalk_loss.append(rwalk_loss.item())
            else:
                loss = seg_loss + adapt_loss_bbox + adapt_loss_samples

            epoch_bbox_loss.append(adapt_loss_bbox.item())
            epoch_samples_loss.append(adapt_loss_samples.item())
            epoch_seg_loss.append(seg_loss.item())

            epoch_loss.append(loss.item())

            # Backpropagate and optimize
            # optimizer.zero_grad()
            if not freeze_all:  # Only when not frozen
                optimizer_model.zero_grad()
            optimizer_adapter.zero_grad()
            loss.backward()
            # optimizer.step()
            if not freeze_all:  # Only when not frozen
                optimizer_model.step()
            optimizer_adapter.step()

            # Calculate the Dice and print it to console --> maybe a log file as well?
            y_pred_ = torch.sigmoid(y_pred_)
            y_pred_ = y_pred_.detach().cpu().numpy().squeeze()
            y_pred_ = (y_pred_ > 0.5).astype(np.uint8)
            y_pred_ = get_one_hot(y_pred_, nb_classes)
            segs = get_one_hot(segs.detach().cpu().numpy().squeeze().astype(np.uint8), nb_classes)
            
            dice_ = DiceMetric(include_background=False, ignore_empty=False)(y_pred_, segs)
            dice.append(np.mean(dice_.numpy())*100)

            # Get compute time
            epoch_step_time.append(time.time() - step_start_time)

            # Create first prev_param after first step as its None otherwise
            if prev_param is None and epoch == 0:
                if not freeze_all:
                    prev_param = {k: torch.clone(v).detach().cpu() for m in [model, adapter] for k, v in m.named_parameters() if v.grad is not None}
                else:
                    prev_param = {k: torch.clone(v).detach().cpu() for m in [adapter] for k, v in m.named_parameters() if v.grad is not None}

        _, _, _, val_res, _ = validate_adapter(model,adapter, val_files, epoch, store_samples=False, out_=os.path.join(args.model_dir, "validation"))

        wandb_log["Train Loss"] = np.mean(epoch_loss)
        wandb_log["Train Seg Loss"] = np.mean(epoch_seg_loss)
        wandb_log["Train Bbox Loss"] = np.mean(epoch_bbox_loss)
        wandb_log["Train Samples Loss"] = np.mean(epoch_samples_loss)
        wandb_log["Train EWC Loss"] = np.mean(epoch_rwalk_loss)
        wandb_log["Train Dice"] = np.mean(dice)
        wandb_log["Val Dice"] = val_res.loc[:, 'Dice'].mean() # val_res["Dice"]
        wandb_log["Val IoU"] = val_res.loc[:, 'IoU'].mean() # val_res["IoU"]

        wandb.log(wandb_log)

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_step_info = '%.4f sec/step' % np.mean(epoch_step_time)
        time_epoch_info = '%.4f sec/epoch' % np.sum(epoch_step_time)
        loss_info = 'train-loss: %.4e' % (np.mean(epoch_loss))
        dice_info = 'train-dice: %.4f' % (np.mean(dice))
        val_dice_info = 'val-dice: %.4f' % (val_res.loc[:, 'Dice'].mean())
        print(' - '.join((epoch_info, time_step_info, time_epoch_info, loss_info, dice_info, val_dice_info)), flush=True)

        # Update best validation dice and model checkpoint if improvement is found
        if val_res.loc[:, 'Dice'].mean() > best_val_dice:
            best_val_dice = val_res.loc[:, 'Dice'].mean()
            model.save(os.path.join(model_dir, 'best_model.pt'))  # Save the best model
            adapter.save(os.path.join(model_dir, 'best_adapter.pt'))
            counter = 0  # Reset the counter as there's improvement
        else:
            counter += 1  # Increment counter if no improvement

        # Check for early stopping
        if counter >= patience:
            print(f"Early stopping at epoch {epoch} as validation dice didn't improve.")
            break  # Exit the training loop

    tot_params, train_params = get_nr_parameters(model)
    model_size = get_model_size(model)
    print("Nr of parameter (total -- trainable): {} -- {}".format(tot_params, train_params))
    print("Model size in MB: {:.4f}".format(model_size))
    tot_params, train_params = get_nr_parameters(adapter)
    model_size = get_model_size(adapter)
    print("Nr of parameter (total -- trainable): {} -- {}".format(tot_params, train_params))
    print("Model size in MB: {:.4f}".format(model_size))

    # final model save and validate
    model.save(os.path.join(model_dir, '%04d_model_final.pt' % args.epochs))
    adapter.save(os.path.join(model_dir, '%04d_adapter_final.pt' % args.epochs))

    # validate_adapter(model, val_files, args.epochs, store_samples=False, out_=os.path.join(args.model_dir, "validation"))
    print("")


    # -- Extract fisher und param values -- #
    model.train()
    if not freeze_all:  # Only when not frozen
        optimizer_model.zero_grad()
    optimizer_adapter.zero_grad()
    for _ in range(args.steps_per_epoch):
        imgs, segs, embeds, orig_sizes, input_sizes, bboxs_gt_, samples_gt_ = next(generator)
        imgs = torch.from_numpy(imgs).to(device).float()    # (n, 256, 256, 3)
        segs = torch.from_numpy(segs).to(device).float()    # (n, 1, 256, 256)
        embeds = torch.from_numpy(embeds).to(device).float()    # (n, 1, 256, 64, 64)
        bboxs_gt = torch.from_numpy(bboxs_gt_).to(device).float()    # (n, 4)
        samples_gt = torch.from_numpy(samples_gt_).to(device).float()    # (n, nr_samples, 2)

        loss = 0
        loss_list = []

        samples_adapt, bbox_adapt = adapter(embeds)
        # imgs = imgs.permute(0, 3, 1, 2)
        # samples_adapt, bbox_adapt = adapter(imgs)

        # Run inputs through the model
        y_pred_, samples_coords_for_loss, bbox_coords_for_loss, _, _ = model(imgs, embeds, orig_sizes, input_sizes, bbox_adapt, samples_adapt, train=True, freeze_all=freeze_all)

        # Flatten the GT samples from [N, nr_samples, 2] --> [n, 2*nr_samples]
        samples_gt_flat = torch.flatten(samples_gt, start_dim=1)    # Size([n, 2*nr_samples])
        
        # Perform bboxs into binary mask to calculate the Dice as well
        seg_mask_size = tuple(segs.squeeze().size())
        nb_classes = segs.size(1) + 1 # Don't forget the background)
        bbox_mask_gt = torch.from_numpy(get_mask_from_bbox_coord(bboxs_gt, seg_mask_size)).to(device).unsqueeze(1) # Size([N, 1, 256, 256])
        
        # Build true and pred lists for loss calculation with Dice
        y_pred = y_pred_
        y_true = segs.long()
        
        # Calculate total loss
        seg_loss = seg_criterion(y_pred, y_true)
        adapt_loss_bbox = prompt_criterion(bbox_adapt, bboxs_gt)
        adapt_loss_samples = prompt_criterion(samples_adapt, samples_gt)

        # loss = seg_criterion(y_pred, y_true)
        if len([x for x in fisher.keys() if x != args.task]) > 0:
            rwalk_loss = rwalk_citerion.loss(adapter.named_parameters())
            if not freeze_all:
                rwalk_loss += rwalk_citerion.loss(model.named_parameters())
            loss = seg_loss + adapt_loss_bbox + adapt_loss_samples + rwalk_loss
            epoch_rwalk_loss.append(rwalk_loss.item())
        else:
            loss = seg_loss + adapt_loss_bbox + adapt_loss_samples

        epoch_bbox_loss.append(adapt_loss_bbox.item())
        epoch_samples_loss.append(adapt_loss_samples.item())
        epoch_seg_loss.append(seg_loss.item())
        epoch_loss.append(loss.item())

        # Backpropagate and optimize
        # optimizer.zero_grad()
        if not freeze_all:  # Only when not frozen
            optimizer_model.zero_grad()
        optimizer_adapter.zero_grad()
        loss.backward()


    # -- Update the importance score one last time once finished training using distance in Riemannian Manifold -- #
    for m in [model, adapter] if not freeze_all else [adapter]:
        for name, param in m.named_parameters():
            if param.grad is not None:
                # -- Get parameter difference from old param and current param t -- #
                delta = param.grad.detach() * (prev_param[name].to(param.device) - param.detach())
                delta = delta.to(0)
                # -- Calculate score denominator -- #
                den = 0.5 * fisher[args.task][name] * (param.detach() - prev_param[name].to(param.device)).pow(2).to(0) + EPSILON
                # -- Score: delat(L) / 0.5*F_t*delta(param)^2 --> only positive or zero values -- #
                scores_ = (delta / den)
                scores_[scores_ < 0] = 0  # Ensure no negative values
                # -- Update the scores -- #
                scores[args.task][name] += scores_

        # -- Store params -- #
        for name, param in m.named_parameters():
            # -- Update the params dict -- #
            params[args.task][name] = param.data.clone()

        # -- Update the fisher values -- #
        for name, param in m.named_parameters():
            # -- F_t = alpha * F_t + (1-alpha) * F_t-1
            if param.grad is not None:
                f_t = param.grad.data.clone().pow(2).to(0)
                f_to = fisher[args.task][name] if fisher[args.task][name] is not None else torch.tensor([0], device='cuda:0')
                fisher[args.task][name] = (ALPHA * f_t) + ((1 - ALPHA) * f_to)

    # -- Normalize the fisher values to be in range 0 to 1 -- #
    values = [torch.max(val) for val in scores[args.task].values()] # --> only for the current task of course
    minim, maxim = min(values), max(values)
    for k, v in fisher[args.task].items():
        fisher[args.task][k] = (v - minim) / (maxim - minim + EPSILON)

    # -- Normalize the score values to be in range 0 to 1 -- #
    values = [torch.max(val) for val in scores[args.task].values()] # --> only for the current task of course
    minim, maxim = min(values), max(values)
    
    if len([x for x in scores.keys() if x != args.task]) > 0:
        # -- Average current and previous scores -- #
        prev_scores = {k: v.clone() for k, v in scores[list(scores.keys())[-1]].items()}
        for k, v in scores[args.task].items():
            # -- Normalize the score -- #
            curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
            # -- Average the score to alleviate rigidity due to the accumulating sum of the scores otherwise -- #
            scores[args.task][k] = 0.5 * (prev_scores[k] + curr_score_norm)
    else:
        # -- Only average current scores -- #
        for k, v in scores[args.task].items():
            # -- Normalize and scale the score so that division does not have an effect -- #
            curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
            scores[args.task][k] = 2 * curr_score_norm

    write_pickle(fisher, os.path.join(model_dir, 'fisher.pkl'))
    write_pickle(params, os.path.join(model_dir, 'params.pkl'))
    write_pickle(scores, os.path.join(model_dir, 'scores.pkl'))


# -- Main function for setup execution -- #
def main():
    train()

if __name__ == "__main__":
    train()