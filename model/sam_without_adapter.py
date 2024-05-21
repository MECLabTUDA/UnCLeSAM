import numpy as np
import random, torch
import skimage.measure
from collections import Counter
from model.sam_adapters import ResNet50Adapter
from segment_anything import sam_model_registry
from model.utils import extract_samples_from_tensor, get_grid
from .modelio import LoadableModel, store_config_args
from segment_anything.utils.transforms import ResizeLongestSide

class Sam_Without_Adapter(LoadableModel):
    """Med_Sam network without adapter for medical imaging.
    """
    @store_config_args
    def __init__(self, inshape, nr_samples, model_type="vit_b", checkpoint="checkpoint/sam_vit_b.pth", device='cpu', **kwargs):
        super(Sam_Without_Adapter, self).__init__()
        self.device = device
        
        self.inshape = inshape
        self.nr_samples = nr_samples

        # -- Initialize and load SAM -- #
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
    
    def forward(self, x, x_embed, orig_size, input_size, bboxes_gt, samples_gt, train=True, freeze_all=False, **kwargs):
        r"""
            Forward pass.
            param x: input image RGB Size([N, 256, 256, 3])
            param x_embed: input image embeding already done with SAM (preprocessing)
            param orig_size: Original size of input image
            param input_size: Size of transformed image
            param train: Set this flag to specify if the call is made during training or not, depending on that the segmentation will be returned differently
            param bboxes_gt: Bounding box coordinates (can be derived from the GT masks)
            param samples_gt: Points (can be obtained from GT masks)
            param freeze_all: Freeze also the seg head during training (for CL methods)
        """
        segs, samples_coords_loss, bbox_coords_loss, samples_coords_plot, bbox_coords_plot = list(), list(), list(), list(), list()
        
        for i, slice in enumerate(x):   
            # -- Build prompt -- #
            adapt_in = slice.permute(2, 0, 1).unsqueeze(0)  # Size([1, 3, 256, 256])
            samples_ = samples_gt[i]
            bbox_ = bboxes_gt[i]
            
            # if freeze_adapter:
            #     with torch.no_grad():
            #         samples_, bbox_, bbox_masks_ = self.Adapter(adapt_in)
            # else:
            #     samples_, bbox_, bbox_masks_ = self.Adapter(adapt_in)
            
            samples_coords_loss.append(samples_) # They are torch and with required_grads for loss calculation
            bbox_coords_loss.append(bbox_) # They are torch and with required_grads for loss calculation
            
            # bbox_masks.append(bbox_masks_) # They are torch and with required_grads for loss calculation
            
            # Create samples into pairs
            input_points = samples_.detach().cpu().numpy().squeeze()  # extract_samples_from_tensor(samples_, self.nr_samples)   # Size([nr_samples, 2])
            input_labels = np.array([1]*self.nr_samples)  # Corresponding label for the points --> since binary for now, always 1
            bbox = bbox_.detach().cpu().numpy().squeeze()   # Size([4])
        
            with torch.no_grad():
                # -- Set the slice in SAM manually using the already extracted embedding -- #
                self.features = x_embed[i].clone()
                self.original_size = orig_size[0]
                self.input_size = input_size[0]
                self.is_image_set = True
                
                # -- Do SAM predict -- #
                # -- Sampled points -- #
                point_coords = self.transform.apply_coords(input_points, self.original_size)
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                labels_torch = torch.as_tensor(input_labels, dtype=torch.int, device=self.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                # -- Bounding Box -- #
                box = self.transform.apply_boxes(bbox, self.original_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
                box_torch = box_torch[None, :]

                # Embed prompts
                if train:
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=(coords_torch, labels_torch),
                        # boxes=None,
                        boxes=box_torch,
                        masks=None,
                    )
                else:
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=(coords_torch, labels_torch),
                        boxes=box_torch,
                        masks=None,
                    )
            
            if freeze_all:
                with torch.no_grad():   # Seg head has gradients
                    # Predict masks
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings = self.features,
                        image_pe = self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings = sparse_embeddings,
                        dense_prompt_embeddings = dense_embeddings,
                        multimask_output = False,
                    )
            else:   # Seg head has NO gradients
                # Predict masks
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings = self.features,
                    image_pe = self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = False,
                )
            
            segs.append(low_res_masks[0])
            # -- For plotting purposes used only
            samples_coords_plot.append(torch.from_numpy(input_points)) # They are torch and for plotting
            bbox_coords_plot.append(torch.from_numpy(bbox)) # They are torch and for plotting
        
        segs = torch.stack(segs, dim=0).float() # Size([N, 1, 256, 256])
        samples_coords_loss = torch.stack(samples_coords_loss, dim=0).float().squeeze()   # Size([N, 200])
        bbox_coords_loss = torch.stack(bbox_coords_loss, dim=0).float().squeeze() # Size([N, 4])
        
        # bbox_masks = torch.stack(bbox_masks, dim=0).float().squeeze() # Size([N, 2, 256, 256])

        # -- For plotting purposes used only
        samples_coords_plot = torch.stack(samples_coords_plot, dim=0).float().squeeze()   # Size([N, 200])
        bbox_coords_plot = torch.stack(bbox_coords_plot, dim=0).float().squeeze() # Size([N, 4])

        if not train:
            segs = torch.sigmoid(segs)
            segs = segs.detach().cpu().numpy().squeeze()
            segs = (segs > 0.5).astype(np.uint8)

        return segs, samples_coords_loss, bbox_coords_loss, samples_coords_plot, bbox_coords_plot # , bbox_masks