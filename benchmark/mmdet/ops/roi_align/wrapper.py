# coding=utf-8
import torch
import torchvision


def roi_align(features, rois, out_size, spatial_scale, sample_num=0):
    return torchvision.ops.roi_align(features, rois, out_size, spatial_scale=spatial_scale)


class RoIAlign(torch.nn.Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RoIAlign, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return torchvision.ops.roi_align(features, rois, self.out_size, spatial_scale=self.spatial_scale)
