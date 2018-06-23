import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Variable

import faster_rcnn.network
from faster_rcnn.network import FC, Conv2d


class IterativeStructure(Module):
    '''
    :param: s_feature: Variable size: batch * 512
    :param: r_feature: Variable size: batch * 512
    :param: o_feature: Variable size: batch * 512
    '''
    def __init__(self, nhidden, dropout):
        super(IterativeStructure, self).__init__()
        self.linear_s_p_o = FC(nhidden, 128, relu=True)
        self.fc_p = FC(768, nhidden, relu=True)
        self.linear_p = FC(nhidden, 128, relu=True)
        self.linear = FC(640, nhidden, relu=True)

    def forward(self, s_feature, r_feature, o_feature):
        s = self.linear_s_p_o(s_feature)  # [batch, 128]
        o = self.linear_s_p_o(o_feature)  # [batch, 128]
        cat_r = torch.cat((s, r_feature, o), 1)
        out_r = self.fc_p(cat_r)   # [batch, 512]

        r = self.linear_p(r_feature)  # [batch, 128]
        cat_s = torch.cat((s_feature, r), 1)   # [batch, 640]
        out_s = self.linear(cat_s)
        cat_o = torch.cat((r, o_feature), 1)
        out_o = self.linear(cat_o)

        return out_s, out_r, out_o


class ConcatEmbedding(Module):
    '''
    :param: s_feature: Variable size: batch * 512
    :param: r_feature: Variable size: batch * 512
    :param: o_feature: Variable size: batch * 512
    '''
    def __init__(self, nhidden, dropout):
        super(ConcatEmbedding, self).__init__()
        self.linear = nn.Linear(3 * nhidden, nhidden)
        self.fc1 = FC(nhidden, 256, relu=True)
        self.fc2 = FC(256, 128, relu=True)
        self.fc3 = FC(128, 128, relu=True)
        self.compress = FC(640, nhidden, relu=True)
        self.compress_r = FC(640, nhidden, relu=True)
        self.dropout = dropout

    def forward(self, s_feature, r_feature, o_feature):
        concat = torch.cat((s_feature, r_feature, o_feature), 1)  # [batch, 1536]
        embed = self.linear(F.relu(concat))  # [batch, 512]
        embed = self.fc1(embed)  # [batch, 256]
        if self.dropout:
            embed = F.dropout(embed, training=self.training)
        embed = self.fc2(embed)  # [batch, 128]
        if self.dropout:
            embed = F.dropout(embed, training=self.training)
        embed = self.fc3(embed)  # [batch, 128]
        if self.dropout:
            embed = F.dropout(embed, training=self.training)

        cat_s = torch.cat((s_feature, embed), 1)  # [batch, 640]
        out_s = self.compress(cat_s)  # [batch, 512]
        if self.dropout:
            out_s = F.dropout(out_s, training=self.training)

        cat_r = torch.cat((r_feature, embed), 1)
        out_r = self.compress_r(cat_r)
        if self.dropout:
            out_r = F.dropout(out_r, training=self.training)

        cat_o = torch.cat((o_feature, embed), 1)
        out_o = self.compress(cat_o)
        if self.dropout:
            out_o = F.dropout(out_o, training=self.training)

        return out_s, out_r, out_o


class Concat(Module):
    '''
    :param: s_feature: Variable size: batch * 512
    :param: r_feature: Variable size: batch * 512
    :param: o_feature: Variable size: batch * 512
    '''
    def __init__(self, nhidden, dropout):
        super(Concat, self).__init__()
        self.conv1 = Conv2d(3 * nhidden, nhidden, 1)
        self.conv2 = Conv2d(nhidden, 256, 1)
        self.conv3 = Conv2d(256, 128, 1)
        self.compress = FC(640, nhidden, relu=True)
        self.compress_r = FC(640, nhidden, relu=True)
        self.dropout = dropout

    def forward(self, s_feature, r_feature, o_feature):
        concat = torch.cat((s_feature, r_feature, o_feature), 1)  # [batch, 1536]
        concat = concat.unsqueeze(2).unsqueeze(3)  # [batch, 1536, 1, 1]
        embed = self.conv1(concat)
        embed = self.conv2(embed)
        embed = self.conv3(embed)   # [batch, 128, 1, 1]
        embed = embed.squeeze(3).squeeze(2)   # [batch, 128]

        cat_s = torch.cat((s_feature, embed), 1)  # [batch, 640]
        out_s = self.compress(cat_s)  # [batch, 512]
        if self.dropout:
            out_s = F.dropout(out_s, training=self.training)

        cat_r = torch.cat((r_feature, embed), 1)
        out_r = self.compress_r(cat_r)
        if self.dropout:
            out_r = F.dropout(out_r, training=self.training)

        cat_o = torch.cat((o_feature, embed), 1)
        out_o = self.compress(cat_o)
        if self.dropout:
            out_o = F.dropout(out_o, training=self.training)

        return out_s, out_r, out_o


class BrnnStructure(Module):
    '''
    :param: s_feature: Variable size: batch * 512
    :param: r_feature: Variable size: batch * 512
    :param: o_feature: Variable size: batch * 512
    '''
    def __init__(self, nhidden, dropout):
        super(BrnnStructure, self).__init__()
        self.rnn = nn.RNN(nhidden, nhidden, 1, bidirectional=True)
        self.compress = FC(3 * nhidden, nhidden, relu=True)
        self.compress_r = FC(3 * nhidden, nhidden, relu=True)
        self.dropout = dropout

    def forward(self, s_feature, r_feature, o_feature):
        s_feature_ = s_feature.unsqueeze(0)  # [1,batch,512]
        r_feature_ = r_feature.unsqueeze(0)
        o_feature_ = o_feature.unsqueeze(0)
        input = torch.cat((s_feature_, r_feature_, o_feature_), 0)  # [3,batch,512]
        _, hn = self.rnn(input)  # [num_layer*num_direction(2), batch, 512]
        hn = hn.transpose(0,1)
        hn = hn.contiguous().view(hn.size()[0], -1)   # [batch, 1024]
        cat_s = torch.cat((s_feature, hn), 1)
        cat_r = torch.cat((r_feature, hn), 1)
        cat_o = torch.cat((o_feature, hn), 1)
        output_s = self.compress(cat_s)
        if self.dropout:
            output_s = F.dropout(output_s, training=self.training)
        output_o = self.compress(cat_o)
        if self.dropout:
            output_o = F.dropout(output_o, training=self.training)
        output_r = self.compress_r(cat_r)
        if self.dropout:
            output_r = F.dropout(output_r, training=self.training)

        return output_s, output_r, output_o


class LstmStructure(Module):
    '''
    :param: s_feature: Variable size: batch * 512
    :param: r_feature: Variable size: batch * 512
    :param: o_feature: Variable size: batch * 512
    '''
    def __init__(self, nhidden, dropout):
        super(LstmStructure, self).__init__()
        self.brnn = nn.LSTM(nhidden, nhidden, 1, bidirectional=True)
        self.compress = FC(3 * nhidden, nhidden, relu=True)
        self.compress_r = FC(3 * nhidden, nhidden, relu=True)
        self.dropout = dropout

    def forward(self, s_feature, r_feature, o_feature):
        s_feature_ = s_feature.unsqueeze(0)  # [1,batch,512]
        r_feature_ = r_feature.unsqueeze(0)
        o_feature_ = o_feature.unsqueeze(0)
        input = torch.cat((s_feature_, r_feature_, o_feature_), 0)  # [3,batch,512]
        hn = self.brnn(input)  # [num_layer*num_direction(2), batch, 512]
        hn = hn[1][0].transpose(0,1)
        hn = hn.contiguous().view(hn.size()[0],-1)   # [batch, 1024]
        cat_s = torch.cat((s_feature, hn), 1)
        cat_r = torch.cat((r_feature, hn), 1)
        cat_o = torch.cat((o_feature, hn), 1)
        output_s = self.compress(cat_s)
        if self.dropout:
            output_s = F.dropout(output_s, training=self.training)
        output_o = self.compress(cat_o)
        if self.dropout:
            output_o = F.dropout(output_o, training=self.training)
        output_r = self.compress_r(cat_r)
        if self.dropout:
            output_r = F.dropout(output_r, training=self.training)

        return output_s, output_r, output_o


class TranslationEmbedding(Module):
    '''
    :param: s_feature: Variable size: batch * 512
    :param: r_feature: Variable size: batch * 512
    :param: o_feature: Variable size: batch * 512
    res: r_feature: Variable size: batch * 512
    '''
    def __init__(self, nhidden, dropout):
        super(TranslationEmbedding, self).__init__()
        self.linear_s = nn.Linear(nhidden, nhidden)
        self.linear_o = nn.Linear(nhidden, nhidden)
        self.fc = FC(2 * nhidden, nhidden, relu=True)
        self.dropout = dropout

    def forward(self, s_feature, r_feature, o_feature):
        trans_s = self.linear_s(s_feature)
        trans_o = self.linear_o(o_feature)
        trans_r = trans_o - trans_s   # [batch, 512]
        r_feature = torch.cat((r_feature, trans_r), 1)  # [batch, 1024]
        r_feature = self.fc(r_feature)
        if self.dropout:
            r_feature = F.dropout(r_feature, training=self.training)

        return r_feature


class GraphicalModel(Module):
    '''
    :param: s_feature: Variable size: batch * 512
    :param: r_feature: Variable size: batch * 512
    :param: o_feature: Variable size: batch * 512
    '''
    def __init__(self, nhidden, dropout):
        super(GraphicalModel, self).__init__()
        self.f = FC(nhidden, nhidden, relu=True)
        self.g = FC(nhidden, nhidden, relu=True)

        self.conv1 = Conv2d(3 * nhidden, nhidden, 1)
        self.conv2 = Conv2d(nhidden, 256, 1)
        self.conv3 = Conv2d(256, 128, 1)
        self.conv1_r = Conv2d(3 * nhidden, nhidden, 1)
        self.conv2_r = Conv2d(nhidden, 256, 1)
        self.conv3_r = Conv2d(256, 128, 1)
        self.compress = FC(640, nhidden, relu=True)
        self.compress_r = FC(640, nhidden, relu=True)
        self.dropout = dropout

    def forward(self, s_feature, r_feature, o_feature):
        s = self.f(s_feature)   # [batch, 512]
        o = self.f(o_feature)
        s_new = s_feature * s   # [batch, 512]
        o_new = o_feature * o

        cat_r = torch.cat((s_new, r_feature, o_new), 1)  # [batch, 1536]
        cat_r = cat_r.unsqueeze(2).unsqueeze(3)  # [batch, 1536, 1, 1]
        embed_r = self.conv1_r(cat_r)
        embed_r = self.conv2_r(embed_r)
        embed_r = self.conv3_r(embed_r)   # [batch, 128, 1, 1]
        embed_r = embed_r.squeeze(3).squeeze(2)   # [batch, 128]

        cat_r = torch.cat((r_feature, embed_r), 1)   # [batch, 640]
        out_r = self.compress_r(cat_r)
        if self.dropout:
            out_r = F.dropout(out_r, training=self.training)

        r = self.g(r_feature)
        r_new = r_feature * r

        cat = torch.cat((s_feature, r_new, o_feature), 1)
        cat = cat.unsqueeze(2).unsqueeze(3)
        embed = self.conv1(cat)
        embed = self.conv2(embed)
        embed = self.conv3(embed)
        embed = embed.squeeze(3).squeeze(2)

        cat_s = torch.cat((s_feature, embed), 1)
        out_s = self.compress(cat_s)
        if self.dropout:
            out_s = F.dropout(out_s, training=self.training)

        cat_o = torch.cat((o_feature, embed), 1)
        out_o = self.compress(cat_o)
        if self.dropout:
            out_o = F.dropout(out_o, training=self.training)

        return out_s, out_r, out_o
