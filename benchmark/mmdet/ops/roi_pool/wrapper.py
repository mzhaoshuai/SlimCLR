# coding=utf-8
import torch
import torchvision


def roi_pool(features, rois, out_size, spatial_scale):
    return torchvision.ops.roi_pool(features, rois, out_size, spatial_scale=spatial_scale)


class RoIPool(torch.nn.Module):

    def __init__(self, out_size, spatial_scale):
        super(RoIPool, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return torchvision.ops.roi_pool(features, rois, self.out_size, spatial_scale=self.spatial_scale)
