import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import dataloading.utils as d_utils
import os, random, torch, copy, cv2, warnings
from monai.metrics import HausdorffDistanceMetric, DiceMetric, MeanIoU, MSEMetric


def set_all_seeds(seed):
  random.seed(seed)
  # os.environ("PYTHONHASHSEED") = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def get_nr_parameters(model):
    r"""This function returns the number of parameters and trainable parameters of a network.
        Based on: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    # -- Extract and count nr of parameters -- #
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # -- Return the information -- #
    return total_params, trainable_params


def get_model_size(model):
    r"""This function return the size in MB of a model.
        Based on: https://discuss.pytorch.org/t/finding-model-size/130275
    """
    # -- Extract parameter and buffer sizes -- #
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    # -- Transform into MB -- #
    size_all_mb = (param_size + buffer_size) / 1024**2
    # -- Return the size -- #
    return size_all_mb

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    res = torch.from_numpy(res)
    return res.reshape(list(targets.shape)+[nb_classes]).permute(0, 3, 1, 2)

def get_mask_from_bbox_coord(coords, shape=(1, 256, 256)):
    r"""
        This function gets 4 coordinates and returns a numpy binary mask, where the area in the bounding box is segmented.
        :param coords: List of 4 coordinates in the form: [N, 4], each coordinate having the format [X1, Y1, X2, Y2]
        :param shape: The final shape of the mask to be returned
    """
    mask = np.zeros(shape, dtype=np.uint8)               # initialize mask
    for idx, coord in enumerate(coords):
        # Calculate the bbox area from coordinates before slicing
        # Xmin, Xmax, Ymin, Ymax = coord
        X1, Y1, X2, Y2 = coord
        # H, W = abs(Y1 - Y2), abs(X1 - X2)
        # mask[idx, int(Ymin):int(Ymax), int(Xmin):int(Xmax)] = 1  # fill with white pixels
        mask[idx, int(Y2):int(Y1), int(X2):int(X1)] = 1  # fill with white pixels, note: always Y2 > Y1 and X2 > X1
        # mask[idx, int(Y1):int(Y1 + H), int(X1):int(X1 + W)] = 1  # fill with white pixels
        # mask[idx, int(coord[2]):int(coord[3]), int(coord[0]):int(coord[1])] = 1  # fill with white pixels
    return mask

def extract_samples_from_tensor(samples_tensor, nr_samples):
    r"""
        Use this to create a list of 2D coordinates (samples) from a tensor resulting in (x, y) coordinate pairs.
    """
    samples_list = list()
    for i in range(nr_samples):
        samples_list.append(samples_tensor[..., i*2:(i+1)*2].detach().cpu().numpy().squeeze())
    return np.asarray(samples_list)


def plot_slice_with_samples_bbox_png(slice, samples, bbox, out, seg=None, neg_samples=None):
    """
    This function takes a slice with sample and bounding box coordinates and stores the image with the
    samples and bbox as a png image using the specified out path.
        :param slice: A 2D RGB slice of an image.
        :param samples: List of samples, i.e. coordinates within the image.
        :param bbox: List of 4 coordinates, i.e. coordinates of a bounding box within the image --> [X1, Y1, X2, Y2].
        :param out: Path to where the slice with plotted samples and bbox should be stored.
    """
    # Build png image
    png = copy.deepcopy(slice)
 
    if seg is not None:
        selected_img = copy.deepcopy(slice)
        colSeg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype="uint8")
        colSeg[:,:,0] = seg * 127 # for red
        colSeg[:,:,1] = seg * 255 # for green
        colSeg[:,:,2] = seg * 212 # for blue
        png = np.uint8(selected_img*0.7 + colSeg*0.3)

    # Make dots for neg_samples red
    if neg_samples is not None:
        for sample in neg_samples:
            try:
                png[int(sample[0]), int(sample[1])] = [255, 0, 0]
            except IndexError:
                pass
                #png[0, 0] = [255, 0, 0]
    
    # Make dots for samples green
    for sample in samples:
        try:
            png[int(sample[0]), int(sample[1])] = [0, 255, 0]
        except IndexError:
            pass
            #png[0, 0] = [0, 255, 0]

    # Make lines for bounding box green
    start_point = (int(bbox[0]), int(bbox[1]))  # represents the top left corner of rectangle
    end_point = (int(bbox[2]), int(bbox[3]))    # represents the bottom right corner of rectangle
    cv2.rectangle(png, start_point, end_point, color=(0,255,0), thickness=1 if slice.shape[1] < 100 else 2)
    
    # Store the image
    io.imsave(out, png, check_contrast=False)

def get_grid(array_size, grid_size):

    # Generate the grid points
    x = np.linspace(0, array_size-1, grid_size)
    y = np.linspace(0, array_size-1, grid_size)

    # Generate a grid of coordinates
    xv, yv = np.meshgrid(x, y)

    # Convert the numpy arrays to lists
    xv_list = xv.tolist()
    yv_list = yv.tolist()

    input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
    input_points = torch.tensor(input_points).view(grid_size*grid_size, 2)

    return input_points