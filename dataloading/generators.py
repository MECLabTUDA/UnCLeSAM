
import numpy as np
import dataloading.utils as d_utils

def npz_generator(vol_names, nr_samples, batch_size=1, **kwargs):
    """
    Generator for npz files.
    
    Parameters:
        path_to_npz_files: Path to npz files to load.
        batch_size: Batch size. Default is 1.
        kwargs:
    """
    if len(vol_names) == 0:
        raise f"I could not find any volumes ending with \'.npz\' in {[v[0] for v in vol_names]}."
    
    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)
        # Load volumes and concatenate
        vols = [d_utils.load_npzfile(vol_names[i], nr_samples) for i in indices]

        # imgs, segs, embeds, orig_sizes, input_sizes = [v[0] for v in vols], [v[1] for v in vols], [v[3] for v in vols], [v[4] for v in vols], [v[5] for v in vols]
        imgs, segs, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt = [v[i] for v in vols for i in range(len(vols[0]))]
        
        # transformed image (b: (n, 256, 256, 3)), segmentation masks (b: (n, 1, 256, 256)), embeddings (b: (n, 1, 256, 64, 64)), bboxs_gt (b: (n, 4)), samples_gt (b: (n, nr_samples, 2))
        yield (imgs, segs, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt)