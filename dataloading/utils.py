import random
import numpy as np


def get_bbox_binary(seg):
    """
    This function calculates a bounding box for a binary image.
    """
    bounding_boxes = []

    seg_idxs = np.argwhere(seg == 1)
    
    if np.any(seg_idxs):    # Only True if array is not empty
        X1 = np.int32(np.min(seg_idxs[:,1]))
        Y1 = np.int32(np.max(seg_idxs[:,1]))
        X2 = np.int32(np.min(seg_idxs[:,0]))
        Y2 = np.int32(np.max(seg_idxs[:,0]))
        
        bounding_boxes = [X1, Y1, X2, Y2]   # Points for cv plotting: (X1, Y1) -- (X2, Y2) being (Xmin, Xmax) -- (Ymin, Ymax), i.e. X1 or Y1 always > X2 or Y2
    
    if len(bounding_boxes) == 0 or not np.any(seg_idxs):
        bounding_boxes = [0, 0, 0, 0]
        
    return bounding_boxes

def load_npzfile(path, nr_samples, **kwargs):
    """
    This function loads a single npz file and returns the values.
    """
    vol = np.load(path)
    orig_sizes = [vol['original_size']]
    input_sizes = [vol['input_size']]
    samples_gt, bboxs_gt = list(), list()

    imgs = [vol['imgs']]
    imgs = np.concatenate(imgs, axis=0)
    segs = [vol['gts'][:, np.newaxis, ...]]
    # find the contours
    for seg in segs:
        for slice in seg:
            x0, x1, y0, y1 = get_bbox_binary(slice.squeeze()) # compute the bounding rectangle of the contour
            bboxs_gt.append([x1, y1, x0, y0,])
            
            if np.sum([x0, y0, x1, y1]) == 0:
                list_empty = []
                for _ in range(nr_samples):
                    list_empty.append([0,0])
                samples_gt.append(list_empty)
            else:
                seg_idxs = np.argwhere(slice == 1)
                # Set a specific random object here so every image get always the same random n points
                myrandom = random.Random(42)
                samples_gt.append([[x[1], x[2]] for x in myrandom.choices(seg_idxs, k=nr_samples)])
        
    bboxs_gt = np.asarray(bboxs_gt)
    samples_gt = np.asarray(samples_gt)
    segs = np.concatenate(segs, axis=0)
    embeds = [vol['img_embeddings']]
    embeds = np.concatenate(embeds, axis=0)

    # transformed image (b: (n, 256, 256, 3)), segmentation masks (b: (n, 1, 256, 256)), embeddings (b: (n, 1, 256, 64, 64)), bboxs_gt (b: (n, 4)), samples_gt (b: (n, nr_samples, 2))
    return (imgs, segs, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt)


def get_bbox_per_cc_binary(seg, jitter=0.):
    """
    This function calculates a bounding box for a binary image by focusing on all CCs.
    If jitter is 0, then the bbox will be close aroung GT, if it is > 0, then the bbox will be larger:
    https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
    """
    bounding_boxes, bbox = [], []
    # NOTE: This extracts for every CC a bounding box, however if this is used, no samples can be used for SAM!
    # Get nr of CCs
    num_labels, labels, _, _ = get_ccs_for_seg(seg)

    # Loop through CCs and get bbox for every CC
    for i in range(1, num_labels):
        seg_idxs = np.argwhere(labels == i)

        if np.any(seg_idxs):    # Only True if array is not empty
            X1 = np.int32(np.min(seg_idxs[:,1]))
            X2 = np.int32(np.max(seg_idxs[:,1]))
            Y1 = np.int32(np.min(seg_idxs[:,0]))
            Y2 = np.int32(np.max(seg_idxs[:,0]))

            # Add jittering, i.e. expand bbox based on jitter value
            X1, X2, Y1, Y2 = np.int32(X1*(1-jitter)), np.int32(X2*(1+jitter)), np.int32(Y1*(1-jitter)), np.int32(Y2*(1+jitter))
                
            bbox = [X1, X2, Y1, Y2]   # Points for cv plotting: (X1, Y1) -- (X2, Y2) being (Xmin, Xmax) -- (Ymin, Ymax), i.e. X1 or Y1 always > X2 or Y2

            bounding_boxes.append(bbox)
        
    return bounding_boxes

def load_niifile(
    filename
):
    """
    Loads a file in nii.gz format which is NOT pre-processed using one of our scripts.

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
    """
    # TODO: In case we did not pre-process but just have the simple nifti image for inference
    # --> or make an inference pre-procesing step here or even script?
    pass