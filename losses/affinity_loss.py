import paddle
from paddle import nn
import paddle.nn.functional as F


class AffinityLoss(nn.Layer):
    
    def __init__(self, num_classes, down_sample_size, reduction='mean', lambda_u=1.0, lambda_g=1.0, align_corners=False):
        super(AffinityLoss, self).__init__()
        self.num_classes = num_classes
        self.down_sample_size = down_sample_size
        if isinstance(down_sample_size, int):
            self.down_sample_size = [down_sample_size] * 2
        self.reduction = reduction
        self.lambda_u = lambda_u
        self.lambda_g = lambda_g
        self.align_corners = align_corners
    
    def forward(self, context_prior_map, label):
        # unary loss
        A = self._construct_ideal_affinity_matrix(label, self.num_classes, self.down_sample_size)
        unary_loss = F.binary_cross_entropy(context_prior_map, A)

        # global loss
        diagonal_matrix = 1 - paddle.eye(A.shape[1])
        vtarget = diagonal_matrix * A
        
        # true intra-class rate(recall)
        recall_part = paddle.sum(context_prior_map * vtarget, axis=2)
        denominator = paddle.sum(vtarget, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        recall_part = recall_part.divide(denominator)
        recall_label = paddle.ones_like(recall_part)
        recall_loss = F.binary_cross_entropy(recall_part, recall_label, reduction=self.reduction)
        
        # true inter-class rate(specificity)
        spec_part = paddle.sum((1 - context_prior_map) * (1 - A), axis=2)
        denominator = paddle.sum(1 - A, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        spec_part = spec_part.divide(denominator)
        spec_label = paddle.ones_like(spec_part)
        spec_loss = F.binary_cross_entropy(spec_part, spec_label, reduction=self.reduction)
        
        # intra-class predictive value(precision)
        precision_part = paddle.sum(context_prior_map * vtarget, axis=2)
        denominator = paddle.sum(context_prior_map, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        precision_part = precision_part.divide(denominator)
        precision_label = paddle.ones_like(precision_part)
        precision_loss = F.binary_cross_entropy(precision_part, precision_label, reduction=self.reduction)

        global_loss = recall_loss + spec_loss + precision_loss 

        return self.lambda_u*unary_loss + self.lambda_g*global_loss

    def _construct_ideal_affinity_matrix(self, label, num_classes, down_sample_size):
        # down sample
        label = paddle.unsqueeze(label, axis=1)
        label = paddle.cast(F.interpolate(label, down_sample_size, mode='nearest', align_corners=self.align_corners), dtype='int64')
        label = paddle.squeeze(label, axis=1)
        label[label == 255] = num_classes

        # to one-hot
        label = F.one_hot(label, num_classes+1)

        # ideal affinity map
        label = paddle.cast(label.reshape((label.shape[0], -1, num_classes+1)), dtype='float32')
        A = paddle.bmm(label, label.transpose((0, 2, 1)))
        return A