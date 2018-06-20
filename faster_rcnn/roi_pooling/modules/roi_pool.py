from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
from ..functions.roi_pool import RoIPoolFunction


class RoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)


class MaskRoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(MaskRoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.pool = torch.nn.AdaptiveMaxPool2d((pooled_height, pooled_width))

    def forward(self, feature_map, rois_1, rois_2):
        """
        :param feature_map: Variable, size: 1 * C * H * W
        :param rois_1: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param rois_2: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :return: pooled_features: Variable, size: Batch * C * pooled_H * pooled_W
        """
        def g(a):
            lambda1 = 0.5
            lambda2 = 0.4
            phi = 4
            return lambda1 + lambda2 * (a**phi)

        def zoom_in_rois(rois, spatial_scale, W, H):
            small_rois = torch.round(rois[:, 1:] * spatial_scale).data.type(torch.cuda.IntTensor)

            over_2 = (small_rois[:, 2] >= W).type(torch.cuda.IntTensor)
            over_0 = (small_rois[:, 0] >= W).type(torch.cuda.IntTensor)
            over_3 = (small_rois[:, 3] >= H).type(torch.cuda.IntTensor)
            over_1 = (small_rois[:, 1] >= H).type(torch.cuda.IntTensor)

            small_rois[:, 0] = small_rois[:, 0] * (1 - over_0) + (W - 1) * over_0
            small_rois[:, 2] = small_rois[:, 2] * (1 - over_2) + (W - 1) * over_2
            small_rois[:, 1] = small_rois[:, 1] * (1 - over_1) + (H - 1) * over_1
            small_rois[:, 3] = small_rois[:, 3] * (1 - over_3) + (H - 1) * over_3

            return small_rois

        def create_mask_map(feature_map):
            feature_map_sum = feature_map.sum(dim=0)
            saliency_map = feature_map_sum / feature_map_sum.max()
            mask_map = g(saliency_map)
            return mask_map

        pooled_features = []
        feature_map = feature_map.squeeze()  # size: C * H * W
        _, H, W = feature_map.data.shape
        mask_map = create_mask_map(feature_map)

        rois_1 = zoom_in_rois(rois_1, self.spatial_scale, W, H)
        rois_2 = zoom_in_rois(rois_2, self.spatial_scale, W, H)
        for roi_1, roi_2 in zip(rois_1, rois_2):
            uni_roi = [min(roi_1[0], roi_2[0]), min(roi_1[1], roi_2[1]),
                       max(roi_1[2], roi_2[2]), max(roi_1[3], roi_2[3])]
            roi_1_rel = [roi_1[0] - uni_roi[0], roi_1[1] - uni_roi[1],
                         roi_1[2] - uni_roi[0], roi_1[3] - uni_roi[1]]
            roi_2_rel = [roi_2[0] - uni_roi[0], roi_2[1] - uni_roi[1],
                         roi_2[2] - uni_roi[0], roi_2[3] - uni_roi[1]]

            uni_roi_feature = feature_map[:, uni_roi[1]:(uni_roi[3]+1), uni_roi[0]:(uni_roi[2]+1)]

            # create mask
            uni_roi_mask = mask_map[uni_roi[1]:(uni_roi[3]+1), uni_roi[0]:(uni_roi[2]+1)].clone()
            # print('uni {}'.format(uni_roi))
            # print('roi1 {}'.format(roi_1.cpu().numpy()))
            # print('roi1rel {}'.format(roi_1_rel))
            # print('roi2 {}'.format(roi_2.cpu().numpy()))
            # print('roi2rel {}'.format(roi_2_rel))
            # print('mask {}'.format(mask.data.shape))
            # print('unimask {}'.format(uni_roi_mask.data.shape))

            uni_roi_mask[roi_1_rel[1]:(roi_1_rel[3]+1), roi_1_rel[0]:(roi_1_rel[2]+1)] = 1
            uni_roi_mask[roi_2_rel[1]:(roi_2_rel[3]+1), roi_2_rel[0]:(roi_2_rel[2]+1)] = 1

            masked_feature = uni_roi_feature * uni_roi_mask

            pooled_features.append(self.pool(masked_feature))
        pooled_features = torch.stack(pooled_features, dim=0)
        return pooled_features


class DualMaskRoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(DualMaskRoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.pool = torch.nn.AdaptiveMaxPool2d((pooled_height, pooled_width))

    def forward(self, feature_map, rois_1, rois_2):
        """
        :param feature_map: Variable, size: 1 * C * H * W
        :param rois_1: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param rois_2: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :return: pooled_features: Variable, size: Batch * C * pooled_H * pooled_W
        """

        def zoom_in_rois(rois, spatial_scale, W, H):
            small_rois = torch.round(rois[:, 1:] * spatial_scale).data.type(torch.cuda.IntTensor)

            over_2 = (small_rois[:, 2] >= W).type(torch.cuda.IntTensor)
            over_0 = (small_rois[:, 0] >= W).type(torch.cuda.IntTensor)
            over_3 = (small_rois[:, 3] >= H).type(torch.cuda.IntTensor)
            over_1 = (small_rois[:, 1] >= H).type(torch.cuda.IntTensor)

            small_rois[:, 0] = small_rois[:, 0] * (1 - over_0) + (W - 1) * over_0
            small_rois[:, 2] = small_rois[:, 2] * (1 - over_2) + (W - 1) * over_2
            small_rois[:, 1] = small_rois[:, 1] * (1 - over_1) + (H - 1) * over_1
            small_rois[:, 3] = small_rois[:, 3] * (1 - over_3) + (H - 1) * over_3

            return small_rois

        pooled_features = []
        feature_map = feature_map.squeeze()  # size: C * H * W
        _, H, W = feature_map.data.shape

        rois_1 = zoom_in_rois(rois_1, self.spatial_scale, W, H)
        rois_2 = zoom_in_rois(rois_2, self.spatial_scale, W, H)
        for roi_1, roi_2 in zip(rois_1, rois_2):
            uni_roi = [min(roi_1[0], roi_2[0]), min(roi_1[1], roi_2[1]),
                       max(roi_1[2], roi_2[2]), max(roi_1[3], roi_2[3])]
            roi_1_rel = [roi_1[0] - uni_roi[0], roi_1[1] - uni_roi[1],
                         roi_1[2] - uni_roi[0], roi_1[3] - uni_roi[1]]
            roi_2_rel = [roi_2[0] - uni_roi[0], roi_2[1] - uni_roi[1],
                         roi_2[2] - uni_roi[0], roi_2[3] - uni_roi[1]]

            uni_roi_feature = feature_map[:, uni_roi[1]:(uni_roi[3] + 1), uni_roi[0]:(uni_roi[2] + 1)]

            # create mask
            uni_roi_mask = Variable(torch.zeros((uni_roi_feature.size()[1], uni_roi_feature.size()[2])).cuda())

            uni_roi_mask[roi_1_rel[1]:(roi_1_rel[3] + 1), roi_1_rel[0]:(roi_1_rel[2] + 1)] = 1
            uni_roi_mask[roi_2_rel[1]:(roi_2_rel[3] + 1), roi_2_rel[0]:(roi_2_rel[2] + 1)] = 1

            masked_feature = uni_roi_feature * uni_roi_mask

            pooled_features.append(self.pool(masked_feature))
        pooled_features = torch.stack(pooled_features, dim=0)
        return pooled_features
