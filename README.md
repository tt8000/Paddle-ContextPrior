# [AI Studio演示](https://aistudio.baidu.com/ibdgpu2/user/517701/2256908)
# 一、Context Prior for Scene Segmentation

## 1. 摘要
	近年来，为了获得更准确的分割结果，人们对上下文相关性进行了广泛的研究。然而，大多数方法很少区分不同类型的上下文依赖关系，这可能会影响场景理解。在这个工作中，我们直接监督特征聚合，以清晰地区分类内上下文和类间上下文。具体来说，我们开发了一个背景先验与亲和损失的监督。在给定输入图像和相应的ground truth的情况下，Affinity Loss构造了一个理想的Affinity map来监督Context Prior的学习。学习的Context Prior提取属于同一类别的像素，而相反的Prior则聚焦于不同类别的像素。本文提出的上下文优先层嵌入到传统的深度CNN中，可以选择性地捕获类内和类间的上下文依赖关系，从而实现鲁棒性特征表示。为了验证有效性，我们设计了一个有效的背景先验网络(CPNet)。广泛的定量和定性评价表明，所提出的模型优于最先进的语义分割方法。更具体地说，我们的算法在ADE20K上实现了46.3% mIoU，在PASCAL-Context上实现了53.9% mIoU，在Cityscape上实现了81.3% mIoU。代码可在https://git.io/ContextPrior上找到。

## 2.  场景分割中的困难例子
![](https://ai-studio-static-online.cdn.bcebos.com/8bcfc3ed8d9c412d843df730b97d0e7554bdbacf6604401a95be8999fbcdb90e)

（1）使用基于金字塔的聚合方法，混淆空间信息的聚合可能导致不良预测。例如：

	在第一行中，红框中沙子的中间部分被误分类为大海，因为阴影部分的外观与大海相似，如(b)所示。

（2）在没有先验知识的情况下，基于注意的方法不能有效地区分混乱的空间信息，导致预测的正确性较差。例如：

	在第二行中，绿色框中的桌子的下半部分被误分类为床，因为其具有与床的底部部分相似的外观，如(e)所示。

（3）在提出在CPNet中，我们将上下文相关性进行了清晰的区分。值得注意的是，Context Prior将类内和类间的关系建模为上下文先验知识，以捕获类内和类间的上下文依赖。例如：

	如(c)，(f)所示。

## 3. 上下文信息聚合方法
（1）基于金字塔的聚合方法：

	其采用金字塔模块或全局池化聚合区域或全局上下文信息。然而，它们捕获了同类上下文关系，忽略了不同类别的上下文依赖关系当场景中存在混淆的类别时，其可能导致不太可靠的上下文。
    
（2）基于注意力的聚合方法：

	目前基于注意的方法学习通道注意、空间注意或点注意来选择性地聚合异质上下文信息。然而，由于缺乏明确的规则化，注意机制的关系描述尚不明确。因此，它可能会选择不需要的上下文依赖关系。

**总的来说，以上方法聚合上下文信息而没有显式的区别，从而导致了不同上下文关系的混合。**

（3）上下文先验(Context Prior区分类内和类间的依赖关系)：

	同一类别之间的相关性(类内上下文)和不同类别之间的差异(类间上下文)使得特征表示更加稳健，减少了可能类别的搜索空间。本文构造上下文先验为一个二值分类器来区分当前那些像素为同一类，那些像素为不同类。

## 4. 本文的主要贡献
（1）我们构建了一个上下文先验的监督嵌入在上下文先验层中的亲和损失(Affinity Loss)，以显式捕获类内(intra-prior)和类间(inter-prior)的上下文依赖。

（2）我们设计了一个有效的场景分割上下文先验网络(CPNet)，该网络包含一个主干网络(Backbone)和一个上下文先验层(Context Prior Layer)。

（3）在ADE20K、Pascal-Context和Cityscape的基准测试中，我们证明了所提出的方法优于最先进的方法。更具体地说，我们的单一模型在ADE20K验证集上达到了46.3%，在PASCALContext验证集上达到了53.9%，在Cityscape测试集上达到了81.3%。

## 5. CPNet

	情景依存关系在情景理解中起着至关重要的作用，各种方法对情景理解进行了广泛的研究。但是，这些方法将不同的上下文依赖聚合为一个混合物。正如第2部分所举例子可以看出，清晰区分的上下文关系对于场景理解是可取的。
    
![](https://ai-studio-static-online.cdn.bcebos.com/3cb6a145369245c9966f687cffb54b8b11b6b097bedf40dd80f9985bc26228c1)

（1）上下文先验层(Context Prior Layer)包括：聚合模块(Aggregation Module)、上下文先验映射( Context
Prior Map)由亲和损失(Affinity Loss)监督。

（2）概述：
> * 利用骨干网络(backbone)得到的输出作为聚合模块的输入聚合空间信息用来推理上下文关系。
> * 生成一个点方向的上下文先验映射，由亲和损失监督(亲和损失：构造一个理想亲和性映射，该映射指示同一类别的像素，以监督上下文先验映射的学习)。
> * 基于上下文先验映射，我们可以得到类内先验(intra-prior：$P$)和类间先验(inter-prior：$1-P$)
> * 将原始特征图($X$)重塑为N × C1大小，其中N = H × W。
> * 使用$P$和$(1-P)$对重塑的特征映射进行矩阵乘法，以捕获类内和类间上下文。
> * 最后，将背景先验层的表示输入到最后一个卷积层，以生成一个逐像素预测。

### 5.1. Affinity Loss
（1）亲和损失

	在场景分割任务中，对于每一幅图像，我们有一个ground truth，它为每个像素分配一个语义类别。网络很难从孤立的像素中建模上下文信息。为了明确正则化网络来建模类别之间的关系，我们引入了亲和损失。对于图像中的每个像素，这种损失迫使网络考虑同一类别的像素(内部上下文)和不同类别之间的像素(内部上下文)。

（2）理想亲和映射(Ideal Affinity Map)
	
    给定输入的ground truth，我们可以知道每个像素的“上下文先验”(即，哪些像素属于同一类别，哪些像素不属于)。因此,我们可以根据ground truth真实情况学习一个上下文先验来指导网络。

![](https://ai-studio-static-online.cdn.bcebos.com/88a529908480465a9501abae38f4f450b7f71b58b4bd4fe0ba3d6b5c5ee0048c)

（3）构建理想亲和映射
> * 给定输入图像$I$和ground truth $L$，将输入图像$I$输入到网络中，得到尺寸为H × W的feature map $X$。
> * 将ground truth $L$向下采样到与$X$的相同大小，得到一个较小的真实值 $\widetilde{L}$。
> * 使用one-hot encoding对ground truth $\widetilde{L}$中的每个类别整数标签进行编码$\hat{L}$，得到矩阵 H × W × C大小，其中C是类的数量。
> * 重塑$\hat{L}$为N × C大小，其中N = H × W。
> * 最后，进行矩阵乘法运算: $A = \hat{L}\hat{L}^{T}$。A 是我们想要的理想亲和映射其大小为N × N，对属于同一类别的像素进行编码。我们使用理想亲和映射来监督上下文先验映射的学习。

（4）亲和损失构建
* 亲和损失包括基于二元交叉熵损失的一元损失(unary loss)和基于二元交叉熵损失的全局损失(global term loss)。
> * $L_{p} = \lambda_{u}L_{u} + \lambda_{g}L_{g}$
> * 其中$L_{p}、L_{u}、L_{g}$代表亲和损失、一元损失、全局损失。
> * $\lambda_{u}$和$\lambda_{g}$分别是一元损失和全局损失的平衡权重。

* 一元损失
> * 对于先验图中的每个像素，都是一个二值分类问题。解决这一问题的传统方法是利用二元交叉熵损失。
> * 给定预测的先验映射$P$，其大小为N × N，其中{${p_{n}∈P, n∈[1,N^{2}]}$}和参考理想亲和力映射$A$，其中{${a_{n}∈A, n∈[1,N^{2}]}$}，二叉熵损失可表示为：
> * $L_{u} = -\frac{1}{N_{2}}\sum_{n=1}^{N^{2}}(a_{n}\log{p_{n}} + (1 - a_{n})\log{(1 - p_{n})})$

* 全局损失
> * 一元损失只考虑了先验映射中孤立的像素，忽略了与其他像素的语义相关性。
> * 先验映射$P$的每一行像素对应feature map $X$的像素，我们可以将它们分为类内像素和类间像素，它们之间的关系有助于推理语义相关和场景结构。
> * 因此，我们可以将类内像素和类间像素视为两个整体，分别对它们之间的关系进行编码。为此，我们设计了基于二元交叉熵损失的全局项：
> * $T_{j}^{p} = \log{\frac{\sum_{i=1}^{N}a_{ij}p_{ij}}{\sum_{i=1}^{N}p_{ij}}}$
> * $T_{j}^{r} = \log{\frac{\sum_{i=1}^{N}a_{ij}p_{ij}}{\sum_{i=1}^{N}a_{ij}}}$
> * $T_{j}^{s} = \log{\frac{\sum_{i=1}^{N}(1 - a_{ij})(1 - p_{ij})}{\sum_{i=1}^{N}(1 - a_{ij})}}$
> * $L_{g} = - \frac{1}{N}\sum_{j=1}^{N}(T_{j}^{p} + T_{j}^{r} + T_{j}^{s})$
> * $T_{j}^{p}$：类内预测值(精度)。
> * $T_{j}^{r}$：真实类内比率(召回)。
> * $T_{j}^{s}$：$P$的第 j 行真实类间比率(特异性)。

### 5.2.  Context Prior Layer
（1）上下文验层考虑一个形状为H × W × C0的输入特征$X$。

（2）采用一个聚合模块使$X$聚合为$\widetilde{X}$，其形状为H × W × C1。

（3）采用一个1 × 1卷积层和一个BN层和一个Sigmoid函数通过$\widetilde{X}$来学习一个大小为H × W × N (N = H × W)的先验映射$P$。
> * 通过对亲和损失的显式监督，上下文先验映射&P&可以对类内像素和类间像素之间的关系进行编码。

（4）内类由$Y = P\widetilde{X}$给出，其中$\widetilde{X}$被重塑为N × C1大小。
> * 在该算子中，先验映射可以自适应地为特征映射中的每个像素选择类内像素作为类内上下文。

（5）另一方面，使用反转的先验映射来选择性地突出类间像素作为类间上下文:$\overline{Y} = (1 - P)\widetilde{X}$，其中1是一个具有相同大小$P$的全1矩阵。

（6）最后，我们将原始特征和两种上下文连接起来，输出最终的预测结果:$F = Concat(X,Y,\overline{Y})$。
> * 通过这两种上下文，我们可以推理每个像素的语义相关性和场景结构。

### 5.3.  Aggregation Module
[李宏毅机器学习进阶-神经网络压缩第4、5部分。](https://aistudio.baidu.com/aistudio/education/group/info/1979)

	上下文先验映射需要一些局部空间信息来推理语义相关性。因此，我们设计了一个具有完全可分离卷积(在空间维度和深度维度上都分离)的有效聚合模块来聚合空间信息。卷积层可以固有地聚合附近的空间信息。一个自然的方法来聚合更多的空间信息是使用一个大的滤波器大小卷积。然而，带有大滤波器尺寸的卷积在计算上是昂贵的。因此，我们在空间上将标准卷积分解为两个非对称卷积。对于k×k卷积，我们可以使用k×1卷积后跟1×k卷积作为替代，称为空间可分离卷积。与标准卷积相比，它可以减少k/2的计算量，并保持接收场大小相等。同时，每个空间可分离卷积都采用深度卷积，进一步导致计算量减少。我们称这种可分离卷积为完全可分离卷积同时考虑了空间维度和深度维度。

![](https://ai-studio-static-online.cdn.bcebos.com/70f4edfcfad84c06923736395db71a7da1dd0390df724fd98e55edfd57cc67ea)

* 聚合模块及其接受域的说明。
> * 我们利用两个不对称的完全可分离卷积来聚合空间信息，其输出与输入特征具有相同的通道。
> * 聚合模块的接收域大小与标准卷积相同。然而，我们的聚合模块导致更少的计算。
> * 符号:Conv标准卷积、DWConv深度卷积、FSConv完全可分卷积、k完全可分卷积的滤波大小、BN批处理归一化、ReLU relu非线性激活函数。

### 5.4. Network Architecture

	上下文先验网络(CPNet)是由骨干网络和上下文先验层组成的全卷积网络，骨干网络是一种现成的卷积网络，如ResNet，具有扩展策略。在上下文先验层中，聚合模块首先有效地聚合一些空间信息。基于聚合的空间信息，上下文先验层学习一个上下文先验映射来获取类内上下文和类间上下文。同时，亲和损失对上下文先验的学习进行了正则化，而交叉熵损失函数则对分割进行了监督。我们在骨干网络的第4阶段采用了辅助损失，这也是一种交叉熵损失。
* 最终损失函数为：
> * $L = \lambda_{s}L_{s} + \lambda_{a}L_{a} + \lambda_{p}L_{p}$
> * 其中$L_{s}、L_{a}、L_{p}$分别代表主要的分割损失函数、辅助损失函数和亲和损失函数。
> * 另外，$\lambda_{s}、\lambda_{a}、\lambda_{p}$分别是平衡分割损失、辅助损失和亲和性损失的权重。

# 二、Code
## 1. 主要代码实现
### 1.1. 安装[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)


```python
!pip -q install paddleseg
```

### 1.2. Affinity_loss 实现


```python
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
```

### 1.3. AggregationModule 实现


```python
import paddle.nn as nn
from paddleseg.models import layers


class AggregationModule(nn.Layer):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(AggregationModule, self).__init__()
        assert isinstance(kernel_size, int), "kernel_size must be a Integer"
        padding = kernel_size // 2
        
        # Conv、BN、ReLu
        self.layer1 = layers.ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1)

        # FSConv
        self.fsconvl = nn.Sequential(
            nn.Conv2D(in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=(1, kernel_size), 
                    padding=(0, padding), 
                    groups=out_channels),
            nn.Conv2D(in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=(kernel_size, 1), 
                    padding=(padding, 0), 
                    groups=out_channels)
        )
        # FSConv
        self.fsconvr = nn.Sequential(
            nn.Conv2D(in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=(kernel_size, 1), 
                    padding=(padding, 0), 
                    groups=out_channels),
            nn.Conv2D(in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=(1, kernel_size), 
                    padding=(0, padding), 
                    groups=out_channels)
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2D(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.layer1(x)
        xl = self.fsconvl(x)
        xr = self.fsconvr(x)
        y = self.layer2(xl + xr)
        return y
```

### 1.4. 辅助 AUXFCHead 实现


```python
import paddle.nn as nn
from paddleseg.models import layers


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
        
        # Conv、BN、ReLu
        self.conv_1 = layers.ConvBNReLU(
            in_channels=backbone_channels[0],
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias_attr=bias)
        # Dropout正则化
        self.dropout = nn.Dropout(drop_out_ratio)
        # 分类
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
```

### 1.5. CPNet 实现


```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
from paddleseg.utils import utils
import numpy as np


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
        
        # 类别
        self.num_classes = num_classes
        # 需要backbone输出的第几阶段
        self.backbone_indices = backbone_indices
        self.prior_channels = prior_channels
        self.prior_size = prior_size
        if isinstance(prior_size, int):
            self.prior_size = [prior_size] * 2
        # 训练模型时的输入大小
        self.size = size
        self.align_corners = align_corners

        # 骨干网络
        self.backbone = backbone
        # 需要backbone输出的第几阶段的输出通道数
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]
        
        # 聚合模块 X -> X^~
        self.aggregation = AggregationModule(backbone_channels[1], 
                                            prior_channels,
                                            am_kernel_size)
        # X^~ -> context prior map
        self.prior_conv = nn.Sequential(
            nn.Conv2D(in_channels=prior_channels,
                    out_channels=np.prod(self.prior_size),
                    kernel_size=1,
                    groups=groups),
            nn.BatchNorm2D(num_features=np.prod(self.prior_size)),
        )
        # 类内上下文学习
        self.intra_conv = layers.ConvBNReLU(
            prior_channels,
            prior_channels,
            kernel_size=1,
            padding=0,
            stride=1
        )
        # 类间上下文学习
        self.inter_conv = layers.ConvBNReLU(
            prior_channels,
            prior_channels,
            kernel_size=1,
            padding=0,
            stride=1
        )
        # 对concat特征进行学习
        self.bottleneck = layers.ConvBNReLU(
            backbone_channels[1]+prior_channels*2,
            channels,
            kernel_size=3,
            padding=1
        )
        # 分类
        self.cls_seg = nn.Sequential(
            nn.Dropout(drop_out_ratio),
            nn.Conv2D(in_channels=channels, out_channels=num_classes, kernel_size=1)
        )
        # 辅助分类
        if enable_auxiliary_loss:
            self.auxlayer = AUXFCHead(
                num_classes=num_classes,
                backbone_indices=(backbone_indices[0], ),
                backbone_channels=(backbone_channels[0], ),
                channels=backbone_channels[0]//4,
                drop_out_ratio=drop_out_ratio
            )
        self.enable_auxiliary_loss = enable_auxiliary_loss
        
        # 初始化
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

        # 分割结果采样到输入图像大小
        logit_list = [
            F.interpolate(
                logit,
                (ori_h, ori_w),
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        
        # [分割结果, 辅助分割结果, 上下文先验映射] or [分割结果, 上下文先验映射]
        logit_list.append(context_prior_map)

        return logit_list
    
    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
```

## 2. 测试
### 2.1. 解压数据集


```python
!rm -rf /home/aistudio/data/ADEChallengeData2016
!pip -q install paddleseg
!unzip -oq /home/aistudio/data/data54455/ADEChallengeData2016.zip -d data/
```

### 2.2. 将数据写入.txt


```python
import os
from tqdm import tqdm


# 将数据写入.txt文件
def data2txt(ori_img_dir, seg_img_dir, txt_path):
    ori_img_path = []
    for line in os.listdir(ori_img_dir):
        ori_img_path.append(line)
    with open(txt_path, 'w') as f:
        for line in tqdm(ori_img_path):
            f.write(f'{ori_img_dir}/{line} {seg_img_dir}/{line.replace("jpg", "png")}\n')


data2txt('/home/aistudio/data/ADEChallengeData2016/images/training',
        '/home/aistudio/data/ADEChallengeData2016/annotations/training',
        '/home/aistudio/train.txt')
data2txt('/home/aistudio/data/ADEChallengeData2016/images/validation',
        '/home/aistudio/data/ADEChallengeData2016/annotations/validation',
        '/home/aistudio/val.txt')
```

### 2.3.切换到Paddle-ContextPrior目录下
* 代码集成到Paddle-ContextPrior目录下，因此到ContextPrior目录下执行代码。


```python
%cd Paddle-ContextPrior/
```

### 2.4. Dataset


```python
import paddle
import paddleseg.transforms as T
from datasets import CPDataset


# 训练数据增强
train_transforms = T.Compose([
    T.ResizeStepScaling(min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25),
    T.RandomHorizontalFlip(),
    T.RandomPaddingCrop(crop_size=(480, 480), label_padding_value=0),
    T.RandomDistort(brightness_range=0.5, contrast_range=0.5, saturation_range=0.5),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 训练数据
train_dataset = CPDataset('/home/aistudio/train.txt', transforms=train_transforms)

# 验证数据增强
eval_transforms = T.Compose([
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 验证数据
val_dataset = CPDataset('/home/aistudio/val.txt', transforms=eval_transforms)
```

### 2.5. 组网


```python
import os
import paddleseg
from paddleseg.models.losses import CrossEntropyLoss
from models.cpnet import CPNet
from losses.affinity_loss import AffinityLoss


num_classes = 150
prior_size = 60

# 模型
model = CPNet(num_classes=num_classes,
            backbone=paddleseg.models.backbones.ResNet50_vd(pretrained='/home/aistudio/ResNet50_vd_pretrained.pdparams'),
            prior_channels=256,
            prior_size=prior_size,
            align_corners=True)

num_steps = 80000
# 优化器
scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=2e-2, decay_steps=num_steps, end_lr=0, power=0.9)
optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4)

# 损失函数
losses = {}
losses['types'] = [
    CrossEntropyLoss(),
    CrossEntropyLoss(),
    AffinityLoss(num_classes=num_classes, down_sample_size=prior_size)
]
# 损失权重
losses['coef'] = [1, 0.4, 1]
```

###  2.6. 训练


```python
from paddleseg.core import train


# 训练
train(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    iters=num_steps,
    batch_size=8,
    save_interval=20000,
    log_iters=100,
    use_vdl=True,
    losses=losses,
    test_config={'aug_eval':True}
)
```

# 三、总结

	本文构建了一个有效的上下文先验的场景分割。它通过监督所提出的亲和损失来区分不同的上下文依赖关系。为了将上下文先验嵌入到网络中，我们提出了一个上下文先验网络，它由主干网络和上下文先验层组成。聚合模块用于聚合空间信息以推理上下文关系，并嵌入到上下文先验层。广泛的定量和定性比较表明，所提出的CPNet优于最近最先进的场景分割方法。
    
（1）对ADE20K验证集的可视化改进。
> * 获取类内上下文和类间上下文有助于场景理解。

![](https://ai-studio-static-online.cdn.bcebos.com/c9ed2c8a20564d6a808fbd626dd06630da21c115e1f945f595abe07bfc2cb824)

（2）CPNet预测的先验地图的可视化。
> * (a)只使用聚合模块来生成注意力地图，而不监督亲和损失。
> * (b)在Affinity Loss的指导下，上下文先验层可以捕获类内上下文和类间上下文。
> * (c)理想亲和关系图是根据ground truth构建的。颜色越深表示响应越高。

![](https://ai-studio-static-online.cdn.bcebos.com/5b60fd66180447cfa96bb1ddde472d30d432a937b28442e3a952caa309f0b0c2)

（3）ADE20K验证集上的分割表现：

![](https://ai-studio-static-online.cdn.bcebos.com/9ef560bf1d394b4aa7269dc285e9bf06ed268ddea5c94f4fac856a2c61bc9731)

（4）PASCAL-Context验证集上的分割表现：

![](https://ai-studio-static-online.cdn.bcebos.com/9910d11b59654e2b88c7537ed41fb24bccf77fdfe6eb404887a108acd2043eb7)

（5）Cityscapes测试集上的分割表现：

![](https://ai-studio-static-online.cdn.bcebos.com/cc5ef39370b64ea88ee809ee1e78e8e24cb59f0695084f4b894c7359f8de7025)

# 个人简介
* 菜鸡一枚~，啥都不会，干饭第一！！！
* [我在AI Studio上获得钻石等级，点亮9个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/517701](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/517701)
