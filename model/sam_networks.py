import numpy as np
import random, torch
import skimage.measure
from collections import Counter
from model.sam_adapters import ResNet50Adapter
from segment_anything import sam_model_registry
from model.utils import extract_samples_from_tensor
from .modelio import LoadableModel, store_config_args
from segment_anything.utils.transforms import ResizeLongestSide

class Own_Sam(LoadableModel):
    """Med_Sam network for medical imaging.
    """
    @store_config_args
    def __init__(self, inshape, nr_samples, model_type="vit_b", checkpoint="checkpoint/sam_vit_b.pth", device='cpu'):
        super(Own_Sam, self).__init__()
        self.device = device
        
        # -- Build the SAM Adapter which is the point cloud generator -- #
        self.Adapter = ResNet50Adapter(nr_samples)
        # self.Adapter = ResNet34Adapter(nr_samples)
        # self.Adapter = LinearAdapter(inshape=inshape, nr_samples=nr_samples)
        # self.Adapter = UNetAdapter(inshape=inshape, nr_samples=nr_samples)
        self.inshape = inshape
        self.nr_samples = nr_samples

        # -- Initialize and load SAM -- #
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
    
    def forward(self, x, x_embed, orig_size, input_size, train=True, freeze_sam=False, freeze_adapter=False, **kwargs):
        r"""
            Forward pass.
            param x: input image RGB Size([N, 256, 256, 3])
            param x_embed: input image embeding already done with SAM (preprocessing)
            param orig_size: Original size of input image
            param input_size: Size of transformed image
            param train: Set this flag to specify if the call is made during training or not, depending on that the segmentation will be returned differently
            param freeze_sam: Set this flag if the SAM segmentation head should be frozen
            param freeze_adapter: Set this flag if the Adapter should be frozen
        """
        segs, samples_coords_loss, bbox_coords_loss, samples_coords_plot, bbox_coords_plot = list(), list(), list(), list(), list()
        for i, slice in enumerate(x):   
            # -- Build prompt -- #
            adapt_in = slice.permute(2, 0, 1).unsqueeze(0)  # Size([1, 3, 256, 256])
            if freeze_adapter:
                with torch.no_grad():
                    samples_, bbox_ = self.Adapter(adapt_in)
            else:
                samples_, bbox_ = self.Adapter(adapt_in)
            samples_coords_loss.append(samples_) # They are torch and with required_grads for loss calculation
            bbox_coords_loss.append(bbox_) # They are torch and with required_grads for loss calculation
            # Create samples into pairs
            input_points = extract_samples_from_tensor(samples_, self.nr_samples)   # Size([nr_samples, 2])
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
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=(coords_torch, labels_torch),
                    # boxes=None,
                    boxes=box_torch,
                    masks=None,
                )

            # Predict masks
            if freeze_sam:
                with torch.no_grad():
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings = self.features,
                        image_pe = self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings = sparse_embeddings,
                        dense_prompt_embeddings = dense_embeddings,
                        multimask_output = False,
                    )
            else:
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

        # -- For plotting purposes used only
        samples_coords_plot = torch.stack(samples_coords_plot, dim=0).float().squeeze()   # Size([N, 200])
        bbox_coords_plot = torch.stack(bbox_coords_plot, dim=0).float().squeeze() # Size([N, 4])

        if not train:
            segs = torch.sigmoid(segs)
            segs = segs.detach().cpu().numpy().squeeze()
            segs = (segs > 0.5).astype(np.uint8)

        return segs, samples_coords_loss, bbox_coords_loss, samples_coords_plot, bbox_coords_plot
    
    def forward_thresholding(self, x, threshold, box_size = 0.05, nr_samples = 5, **kwargs):
        segs = list()

        # -- Do thresholding here -- #
        img_ = np.where(x > threshold, 1., 0.)

        # -- Calculate center and borders -- #
        center_point = [img_.shape[1]//2, img_.shape[2]//2]
        borders_x = [int(center_point[0]+box_size*img_.shape[1]), int(center_point[0]-box_size*img_.shape[1])]
        borders_y = [int(center_point[1]+box_size*img_.shape[2]), int(center_point[1]-box_size*img_.shape[2])]
        input_box = np.array([borders_x[1], borders_y[1], borders_x[0], borders_y[0]])  # xyxy format

        # Thresholded: Mask first so only ofcus on bounding box --> then keep only the most occured label based on CC and take n samples from that
        mask = np.zeros_like(img_) 
        mask = mask==0
        mask[:, borders_x[1]:borders_x[0], borders_y[1]:borders_y[0], :] = False
        img_[mask] = 0
        labeled_image, count = skimage.measure.label(img_[:,:,:,0], return_num=True)
        lab_idxs = labeled_image.flatten().tolist()
        counter = Counter(lab_idxs)
        # max_id = max(set(lab_idxs), key = lab_idxs.count)
        if counter.most_common(1)[0][0] == 0:   # Skip background label
            try:
                max_id = counter.most_common(2)[1][0]
            except:
                max_id = counter.most_common(1)[0][0]
        else:
            max_id = counter.most_common(1)[0][0]
            
        img_ = np.uint8(labeled_image==max_id)
        # img_ = np.where(img_==float(max_id), 1., 0.)  # <-- mig_ already contains only 0 and 1
        img_ = np.uint8(np.repeat(img_[:,:,:,None], 3, axis=-1))
        
        for i, slice in enumerate(img_):
            idxs_ = [y[:2] for y in np.argwhere(slice==1.)]

        # # -- Extract CCs -- #
        # labeled_image, _ = skimage.measure.label(img_[:,:,:,0], return_num=True)
        # slice = img_[img_.shape[0]//2,:,:,:]

        # # -- Pick centwered slice and select nr_samples from withing bounding box -- #
        # keypoints = dict()
        # idxs = [y[:2] for y in np.argwhere(slice!=0)]
        # idxs_ = list()
        # for idx in idxs:
        #     if idx[0] >= borders_x[1] and idx[0] <= borders_x[0] and idx[1] >= borders_y[1] and idx[1] <= borders_x[0]:
        #         idxs_.append(idx)
        # samples = random.sample(idxs_, nr_samples)
        # keypoints['0'] = samples
        
        # # -- Take only the CC with the highest occurence in all slices within the bounding box -->  Still TODO currently only for one slice... -- #
        # # TODO: Don't do it randomly n times but take all labels within the bounding box for every slice and take the most occured label
        # lab_idxs = [labeled_image[labeled_image.shape[0]//2, keypoints['0'][i][0], keypoints['0'][i][1]] for i in range(5)]
        # max_id = max(set(lab_idxs), key = lab_idxs.count)
        
        # img_ = np.uint8(labeled_image==max_id)
        # img_ = np.uint8(np.repeat(img_[:,:,:,None], 3, axis=-1))

        # # -- Sample some points -- #
        # # for slice in img_:
        # for slice in x:
        #     idxs = [y[:2] for y in np.argwhere(slice==1)]
        #     idxs_ = list()
        #     for idx in idxs:
        #         if idx[0] >= borders_x[1] and idx[0] <= borders_x[0] and idx[1] >= borders_y[1] and idx[1] <= borders_x[0]:
        #             idxs_.append(idx)


            # # -- Simple predictions -- #
            # self.set_image(slice)
            # seg, _, _ = self.predict(point_coords=None, box=input_box, multimask_output=False)
            # segs.append(seg[0])


            # H, W = slice.shape[:2]
            # # resize_img = self.transform.apply_image(slice)
            # # resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
            # # input_image = self.sam_model.preprocess(resize_img_tensor[None,:,:,:])    # (1, 3, 1024, 1024)


            # with torch.no_grad():
            #     image_embedding = kwargs['embeds'] #self.sam_model.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
            #     # convert box to 1024x1024 grid
            #     bbox = self.transform.apply_boxes(input_box, (H, W))
            #     box_torch = torch.as_tensor(bbox, dtype=torch.float, device=self.device)
            #     if len(box_torch.shape) == 2:
            #         box_torch = box_torch[:, None, :] # (B, 1, 4)
                
            #     sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            #         points=None,
            #         boxes=box_torch,
            #         masks=None,
            #     )
            #     seg_prob, _ = self.sam_model.mask_decoder(
            #         image_embeddings=image_embedding.to(self.device), # (B, 256, 64, 64)
            #         image_pe=self.sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            #         sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            #         dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            #         multimask_output=False,
            #         )
            #     seg_prob = torch.sigmoid(seg_prob)
            #     # convert soft mask to hard mask
            #     seg_prob = seg_prob.cpu().numpy().squeeze()
            #     seg = (seg_prob > 0.5).astype(np.uint8)
            #     segs.append(seg)



            # -- Build prompt -- #
            try:
                input_points = np.array(random.sample(idxs_, nr_samples))    # List of points that are sampled from the point cloud
                input_labels = np.array([1]*nr_samples)            # Corresponding label for the points
            except:
                segs.append(np.zeros(slice[...,0].shape))
                continue
            
            # # input_point = np.array([[500, 375], [1125, 625]])    # List of points that are sampled from the point cloud
            # # input_label = np.array([1, 0])             # Corresponding label for the points

            # # -- Generate segmentation using SAM -- #
            # # slice = slice[..., np.newaxis]
            # dim = np.zeros(slice.shape)
            # slice = np.stack((slice, dim, dim), axis=2)

            self.set_image(x[i])#.astype(np.uint8))
            masks, _, _ = self.predict(
                                            point_coords=input_points,
                                            point_labels=input_labels,
                                            # box=input_box,
                                            multimask_output=False # if True it returns three masks..
                                            )
            # # seg_x = masks[torch.argmax(scores)]
            segs.append(masks[0])
        
        segs = np.stack(segs, axis=0).astype(int)
        return segs