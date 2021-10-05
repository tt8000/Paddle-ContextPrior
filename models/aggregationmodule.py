import paddle.nn as nn
from paddleseg.models import layers


class AggregationModule(nn.Layer):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(AggregationModule, self).__init__()
        assert isinstance(kernel_size, int), "kernel_size must be a Integer"
        padding = kernel_size // 2

        self.layer1 = layers.ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1)

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