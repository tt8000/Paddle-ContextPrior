import os
import paddle
import paddleseg.transforms as T
import paddleseg
from paddleseg.models.losses import CrossEntropyLoss
from paddleseg.core import train
from datasets import CPDataset
from models.cpnet import CPNet
from losses.affinity_loss import AffinityLoss


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
