import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Variable

import network
from network import FC

class RecurrentAttention(Module):
    '''
    :param: x_s_i: Variable size: Batch * 512
    :param: x_p_j: Variable size: Batch * 512
    :param: x_o_i: Variable size: Batch * 512
    '''
    def __init__(self, nhidden):
        super(RecurrentAttention, self).__init__()
        self.rnn = nn.LSTM(nhidden, nhidden, 1)
        self.rnn_r = nn.LSTM(nhidden, nhidden, 1)

    def forward(self, x_s_i, x_p_j, x_o_i):
        r = x_p_j.unsqueeze(0)   # [1,batch,512]
        s = x_s_i.unsqueeze(0)
        o = x_o_i.unsqueeze(0)
        g = self.rnn(r)[1][0].squeeze(0)   # [batch,512]
        f_s = self.rnn(s)[1][0].squeeze(0)
        f_o = self.rnn(o)[1][0].squeeze(0)
        s_p = torch.matmul(F.softmax(torch.matmul(x_s_i, g.t())), g)
        p_o = torch.matmul(F.softmax(torch.matmul(x_o_i, g.t())), g)
        out_r = (s_p + p_o)/2   # [batch,512]
        out_s = torch.matmul(F.softmax(torch.matmul(x_p_j, f_s.t())), f_s)
        out_o = torch.matmul(F.softmax(torch.matmul(x_p_j, f_o.t())), f_o)

        return out_s, out_r, out_o


class AttentionBilinear(Module):
    def __init__(self):
        super(AttentionBilinear, self).__init__()
        self.bilinear = nn.Bilinear(512, 512, 512)
        self.bilinear_r = nn.Bilinear(512, 512, 512)

    def forward(self, feature1, feature2, feature3):
        '''
        :param feature1: Variable size: Batch * 512
        :param feature2: Variable size: Batch * 512
        :param feature3: Variable size: Batch * 512
        :return:
        '''
        s_feature = self.compute(feature1)
        o_feature = self.compute(feature3)

        similarity_r = self.bilinear_r(feature2, feature2)
        attention_r = F.softmax(similarity_r)
        r_feature = torch.matmul(attention_r, feature2)

        return s_feature, r_feature, o_feature

    def compute(self, x):
        similarity = self.bilinear(x, x)
        attention = F.softmax(similarity)
        feature = torch.matmul(attention, x)

        return feature


def attention(feature1, feature2, feature3):
    '''
    :param feature1: Variable size: Batch * 512
    :param feature2: Variable size: Batch * 512
    :param feature3: Variable size: Batch * 512
    :return:
    '''
    def compute(x):
        '''
        :param x: Variable size: Batch * 512
        :return: feature: Variable size: Batch * 512
        '''
        similarity = torch.matmul(x, x.transpose(0,1))  # [batch, batch]
        attention = F.softmax(similarity)
        feature = torch.matmul(attention, x)  # [batch, 512]

        return feature

    s_feature = compute(feature1)
    r_feature = compute(feature2)
    o_feature = compute(feature3)

    return s_feature, r_feature, o_feature