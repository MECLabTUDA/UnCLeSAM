import numpy as np
from torch import nn
from model.utils import *
from torchvision import models
from model.unet_utils import UNet
from model.pcg_utils import Encoder, Decoder
import torch.nn.functional as F
from .modelio import LoadableModel, store_config_args
from torchvision.models import resnet50, ResNet50_Weights

# TODO: Use samples and create bbox out of that

class UNetAdapter(nn.Module):
    r"""
        Adapter which uses a UNet to predict the bounding box and a resnet for sampling.
    """
    def __init__(self, inshape, nr_samples):
        r"""
            param inshape: inshape of the image slic, e.g. (256, 256, 3)
            param nr_samples: number of samples we want to draw later on
        """
        super().__init__()
        self.unet = UNet(
                            inshape=inshape[:2],    # <-- Use it in 2D
                            infeats=1,  # <-- One input feature
                            nb_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
                            nb_levels=None,
                            feat_mult=1,
                            feat_out=2, # <-- 2 output features (binary)
                            nb_conv_per_level=1,
                            half_res=False
                          )
        self.nr_samples = nr_samples
        feats = np.prod(inshape[:2]) * self.unet.final_nf
        self.sampler = nn.Sequential(nn.InstanceNorm1d(feats), nn.Linear(feats, self.nr_samples*2))
        self.bbox = nn.Sequential(nn.InstanceNorm1d(feats), nn.Linear(feats, 4))

    def forward(self, x):       # Size([1, 3, 256, 256])
        in_size = tuple(x[:, 0, ...].size())    # Size([1, 256, 256])
        bbox_mask = self.unet(x[:, 0, ...].unsqueeze(0))  # Size([1, 64, 64, 16])

        samples = self.sampler(bbox_mask.flatten(start_dim=1)) # Size([1, 2*nr_samples])
        bbox = self.bbox(bbox_mask.flatten(start_dim=1))   # Size([1, 4])

        # samples = self.sampler(bbox_mask.flatten(start_dim=1).detach()) # Size([1, 2*nr_samples])
        # bbox = self.bbox(bbox_mask.flatten(start_dim=1).detach())   # Size([1, 4])
        # bbox_mask_ = torch.sigmoid(bbox_mask)
        
        # Use bbox_mask from feats, and extract bbox coordinates from it as well
        # Points for cv plotting: (X1, Y1) -- (X2, Y2) being (Xmin, Xmax) -- (Ymin, Ymax), i.e. X1 or Y1 always > X2 or Y2
        # segs = torch.round(bbox_mask_)
        # X1 = torch.min(segs[:,1])
        # Y1 = torch.max(segs[:,1])
        # X2 = torch.min(segs[:,0])
        # Y2 = torch.min(segs[:,0])

        # bbox = torch.tensor([X1, Y1, X2, Y2], requires_grad = True).to(Y2.device)   # Points for cv plotting: (X1, Y1) -- (X2, Y2) being (Xmin, Xmax) -- (Ymin, Ymax), i.e. X1 or Y1 always > X2 or Y2

        return samples, bbox, bbox_mask

class PCGAdapter(nn.Module):
    r"""
        Structure generator components in PCG for SAM.
    """
    def __init__(self, encoder=None, decoder=None,
                 outViewN=8, outW=128, outH=128, renderDepth=1.0):
        super(PCGAdapter, self).__init__()

        if encoder: self.encoder = encoder
        else: self.encoder = Encoder()

        if decoder: self.decoder = decoder
        else: self.decoder = Decoder(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        latent = self.encoder(x)
        XYZ, maskLogit = self.decoder(latent)

        return XYZ, maskLogit

class LinearAdapter(nn.Module):
    r"""
        Simple Linear Regression Adapter for SAM predicting n sample coordinates and a bounding box.
    """
    def __init__(self, inshape, nr_samples):
        r"""
            param inshape: inshape of the image slic, e.g. (256, 256, 3)
            param nr_samples: number of samples we want to draw later on
        """
        super(LinearAdapter, self).__init__()
        self.nr_samples = nr_samples
        self.body = nn.Sequential(
                                    nn.Linear(np.prod(inshape), 2**8),
                                    nn.ReLU(),
                                    # nn.Linear(2**10, 2**8),
                                    # nn.ReLU(),
                                    nn.Linear(2**8, 2**6),
                                    nn.ReLU(),
                                #  nn.Linear(64, nr_samples*2)
                                )

        self.sampler = nn.Sequential(nn.InstanceNorm1d(2**6), nn.Linear(2**6, self.nr_samples*2))
        self.bb = nn.Sequential(nn.InstanceNorm1d(2**6), nn.Linear(2**6, 4))

    def forward(self, x):
        in_size = tuple(x[:, 0, ...].size())    # Size([1, 256, 256])
        feats = self.body(x.flatten(start_dim=1))
        samples = self.sampler(feats) #_extract_samples_from_tensor(self.sampler(feats), self.nr_samples)
        bbox = self.bb(feats)
        # bbox_mask = torch.from_numpy(get_mask_from_bbox_coord(bbox, in_size)).to(x.device).float().requires_grad_()
        return samples, bbox#, bbox_mask

class ResNet34Adapter(nn.Module):
    r"""
        Simple ResNet34 Adapter for SAM predicting n sample coordinates and a bounding box.
    """
    def __init__(self, nr_samples, **kwargs):
        r"""
            param nr_samples: number of samples we want to draw later on
        """
        super(ResNet34Adapter, self).__init__()
        self.nr_samples = nr_samples
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.sampler = nn.Sequential(nn.InstanceNorm1d(512), nn.Linear(512, self.nr_samples*2))
        self.bb = nn.Sequential(nn.InstanceNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):       # Size([1, 3, 256, 256])
        in_size = tuple(x[:, 0, ...].size())    # Size([1, 256, 256])
        x = self.features1(x)   # Size([1, 512, 32, 32])
        x = self.features2(x)   # Size([1, 2048, 8, 8])
        x = nn.functional.relu(x)   # Size([1, 2048, 8, 8])
        x = nn.AdaptiveAvgPool2d((1,1))(x)          # Size([1, 2048, 8, 8])
        feats = x.view(x.shape[0], -1)      # Size([1, 2048])
        samples = self.sampler(feats) #_extract_samples_from_tensor(self.sampler(feats), self.nr_samples)   # Size([1, 2*nr_samples])
        bbox = self.bb(feats)   # Size([1, 4])
        # bbox_mask = torch.from_numpy(get_mask_from_bbox_coord(bbox, in_size)).to(x.device).float().requires_grad_()
        return samples, bbox#, bbox_mask

class ResNet50Adapter(nn.Module):
    r"""
        Simple ResNet50 Adapter for SAM predicting n sample coordinates and a bounding box.
    """
    def __init__(self, nr_samples, **kwargs):
        r"""
            param nr_samples: number of samples we want to draw later on
        """
        super(ResNet50Adapter, self).__init__()
        self.nr_samples = nr_samples
        resnet = models.resnet50()#weights=models.ResNet50_Weights.DEFAULT)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        # self.sampler = nn.Sequential(nn.InstanceNorm1d(2048), nn.Linear(2048, self.nr_samples*2))
        # self.bb = nn.Sequential(nn.InstanceNorm1d(2048), nn.Linear(2048, 4))
        self.final = nn.Sequential(nn.InstanceNorm1d(2048), nn.Linear(2048, self.nr_samples*2 + 4))

    def forward(self, x):       # Size([1, 3, 256, 256])
        in_size = tuple(x[:, 0, ...].size())    # Size([1, 256, 256])
        x = self.features1(x)   # Size([1, 512, 32, 32])
        x = self.features2(x)   # Size([1, 2048, 8, 8])
        x = nn.functional.relu(x)   # Size([1, 2048, 8, 8])
        x = nn.AdaptiveAvgPool2d((1,1))(x)          # Size([1, 2048, 8, 8])
        feats = x.view(x.shape[0], -1)      # Size([1, 2048])
        res = self.final(feats)
        samples, bbox = res[..., :-4], res[..., -4:]
        # samples = self.sampler(feats) #_extract_samples_from_tensor(self.sampler(feats), self.nr_samples)   # Size([1, 2*nr_samples])
        # bbox = self.bb(feats)   # Size([1, 4])
        # bbox_mask = torch.from_numpy(get_mask_from_bbox_coord(bbox, in_size)).to(x.device).float().requires_grad_()
        return samples, bbox#, bbox_mask

class ConvAdapter(LoadableModel):
    @store_config_args
    def __init__(self, nr_samples, **kwargs):
        super(ConvAdapter, self).__init__()
        self.nr_samples = nr_samples
        self.conv_transpose = nn.ConvTranspose2d(256, 3, kernel_size=4, stride=4, padding=0)
        self.resnet50 = resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])  # remove the last fc layer
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2 * nr_samples)
        self.fc3 = nn.Linear(1024, 4)

    def forward(self, x, **kwargs):
        x = self.conv_transpose(x.squeeze())
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        samples = self.fc2(x)
        bbox = self.fc3(x)
        return samples.view(-1, self.nr_samples, 2), bbox.view(-1, 4)