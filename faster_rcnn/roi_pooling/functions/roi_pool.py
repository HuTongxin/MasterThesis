import torch
from torch.autograd import Function
from .._ext import roi_pooling


class RoIPoolFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.output = None
        self.argmax = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()  # features.size: [1,512,W,H]
        num_rois = rois.size()[0]  # num_rois: 32
        output = torch.cuda.FloatTensor(num_rois, num_channels, self.pooled_height, self.pooled_width).zero_()
        argmax = torch.cuda.IntTensor(num_rois, num_channels, self.pooled_height, self.pooled_width).zero_()

        if not features.is_cuda:
            assert False, 'feature not in CUDA'
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(self.pooled_height, self.pooled_width, self.spatial_scale,
                                            _features, rois, output)
            # output = output.cuda()
        else:
            roi_pooling.roi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                 features, rois, output, argmax)
            self.output = output
            self.argmax = argmax
            self.rois = rois
            self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        # TODO: roi_pooling backward

        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()
        roi_pooling.roi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                              grad_output, self.rois, grad_input, self.argmax)

        # print grad_input

        return grad_input, None
