import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg.utils import utils
import numpy as np

from .aggregationmodule import AggregationModule


class AUXFCHead(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None,
                 drop_out_ratio=0.1,
                 bias=False):
        super(AUXFCHead, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        if channels is None:
            channels = backbone_channels[0]

        self.conv_1 = layers.ConvBNReLU(
            in_channels=backbone_channels[0],
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias_attr=bias)
        self.dropout = nn.Dropout(drop_out_ratio)
        self.cls = nn.Conv2D(
            in_channels=channels,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            bias_attr=bias)

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.conv_1(x)
        x = self.dropout(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list


class CPNet(nn.Layer):

    def __init__(self,
                num_classes,
                backbone,
                backbone_indices=[2, 3],
                channels=512,
                prior_channels=512,
                prior_size=64,
                am_kernel_size=11,
                groups=1,
                drop_out_ratio=0.1,
                size=(480, 480),
                enable_auxiliary_loss=True,
                align_corners=False,
                pretrained=None):
        super(CPNet, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        self.prior_channels = prior_channels
        self.prior_size = prior_size
        if isinstance(prior_size, int):
            self.prior_size = [prior_size] * 2
        self.size = size
        self.align_corners = align_corners

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.aggregation = AggregationModule(backbone_channels[1], 
                                            prior_channels,
                                            am_kernel_size)

        self.prior_conv = nn.Sequential(
            nn.Conv2D(in_channels=prior_channels,
                    out_channels=np.prod(self.prior_size),
                    kernel_size=1,
                    groups=groups),
            nn.BatchNorm2D(num_features=np.prod(self.prior_size)),
        )
        self.intra_conv = layers.ConvBNReLU(
            prior_channels,
            prior_channels,
            kernel_size=1,
            padding=0,
            stride=1
        )

        self.inter_conv = layers.ConvBNReLU(
            prior_channels,
            prior_channels,
            kernel_size=1,
            padding=0,
            stride=1
        )
        self.bottleneck = layers.ConvBNReLU(
            backbone_channels[1]+prior_channels*2,
            channels,
            kernel_size=3,
            padding=1
        )
        
        self.cls_seg = nn.Sequential(
            nn.Dropout(drop_out_ratio),
            nn.Conv2D(in_channels=channels, out_channels=num_classes, kernel_size=1)
        )

        if enable_auxiliary_loss:
            self.auxlayer = AUXFCHead(
                num_classes=num_classes,
                backbone_indices=(backbone_indices[0], ),
                backbone_channels=(backbone_channels[0], ),
                channels=backbone_channels[0]//4,
                drop_out_ratio=drop_out_ratio
            )
        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.pretrained = pretrained
        self.init_weight()
        
    def forward(self, x):
        ori_h, ori_w = x.shape[2:]
        if x.shape[2:] != self.size:
            x = F.interpolate(x, self.size,
                        mode='bilinear',
                        align_corners=self.align_corners)
        
        feat_list = self.backbone(x)
        batch_size, channels, height, width = feat_list[self.backbone_indices[1]].shape
        assert self.prior_size[0] == height and self.prior_size[1] == width, \
        'prior_size='+str(self.prior_size)+', height='+str(height)+', width='+str(width)+'; prior_size must equal to [height, width]'

        xt = self.aggregation(feat_list[self.backbone_indices[1]])

        # context prior map
        context_prior_map = self.prior_conv(xt)
        context_prior_map = context_prior_map.reshape((batch_size, np.prod(self.prior_size), -1))
        context_prior_map = context_prior_map.transpose((0, 2, 1))
        context_prior_map = F.sigmoid(context_prior_map)
        
        # xt reshape to BxC1xN -> BxNxC1
        xt = xt.reshape((batch_size, self.prior_channels, -1))
        xt = xt.transpose((0, 2, 1))

        # intra-class context
        intra_context = paddle.bmm(context_prior_map, xt)
        intra_context = intra_context.divide(paddle.to_tensor(np.prod(self.prior_size), dtype='float32'))
        intra_context = intra_context.transpose((0, 2, 1))
        intra_context = intra_context.reshape((batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1]))
        intra_context = self.intra_conv(intra_context)

        # inter-class context
        inter_context_prior_map = 1 - context_prior_map
        inter_context = paddle.bmm(inter_context_prior_map, xt)
        inter_context = inter_context.divide(paddle.to_tensor(np.prod(self.prior_size), dtype='float32'))
        inter_context = inter_context.transpose((0, 2, 1))
        inter_context = inter_context.reshape((batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1]))
        inter_context = self.inter_conv(inter_context)

        # concat
        concat_x = paddle.concat([feat_list[self.backbone_indices[1]], intra_context, inter_context], axis=1)

        # classification
        logit_list = []
        logit_list.append(self.cls_seg(self.bottleneck(concat_x)))

        # aux classification
        if self.enable_auxiliary_loss:
            auxiliary_logit = self.auxlayer(feat_list)
            logit_list.extend(auxiliary_logit)

        
        logit_list = [
            F.interpolate(
                logit,
                (ori_h, ori_w),
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

        logit_list.append(context_prior_map)

        return logit_list
    
    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)