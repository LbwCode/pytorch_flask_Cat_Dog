from torch import nn
import torch.nn.functional as F
import torch
import math

"""
复现vgg19,增加一层实现2分类
"""


class VGG19(nn.Module):
    def __init__(self, cfg=None):
        super(VGG19, self).__init__()
        if cfg is None:
            # 使用Max是证明里面可以随意定义，但在剪枝内要与之对应改动
            cfg = [64, 'Max', 64, 128, 'Max',
                   128, 256, 256, 256, 'Max',
                   256, 512, 512, 512, 'Max',
                   512, 512, 512, 512, 'Max']
        self.vgg19 = self.make_vgg19(cfg)
        self.f1 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # 由于剪枝模型后，nn.Conv2d(in_channels=？？？) 或者叫 输入通道数，也就是配置文件cfg会产生变化
        # 全连接层的输入是由上一层的输出决定的，这里不能定死，要写成一个变量
        # 全连接层根据 nn.AdaptiveAvgPool2d(output_size=(7, 7)) 与 cfg的最后一层，如果最后一层是池化，那就用倒数第二层
        # 全连接层根据 图像走到当前层的宽、高 *  特征图个数 也就是cfg最后的数
        self.f2 = nn.Linear(in_features=cfg[-2]*7*7, out_features=4096, bias=True)
        self.Relu = nn.ReLU(inplace=True)
        self.f3 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.f4 = nn.Linear(in_features=4096, out_features=1000, bias=True)
        self.f5 = nn.Linear(in_features=1000, out_features=2, bias=True)
        # self.f6 = nn.Sigmoid()
        # 权重初始化
        # self._initialize_weights()

    def make_vgg19(self, cfg):
        """
        函数概述：构造vgg19的网络模型，前半部分
        诞生原因：利用自定义的cfg来快速构造重复性较多的层（少写点代码）
        通用具体实现思路：观察网络重复节点，进行打包，按序添加列表中再串联。
        """
        # 传递给下一层输入的节点数
        input_n = 3  # 初始的输入是3
        # 总网络层数
        features_sum = []
        for v in cfg:
            if v == 'Max':
                features_sum += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]
            else:
                conv = nn.Conv2d(input_n, v, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                features_sum += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                input_n = v
        return nn.Sequential(*features_sum)

    def forward(self, x):
        x = self.vgg19(x)
        x = self.f1(x)
        x = nn.Flatten()(x)
        x = self.f2(x)
        x = self.Relu(x)
        x = F.dropout(x, p=0.5)
        x = self.f3(x)
        x = F.dropout(x, p=0.5)
        x = self.f4(x)
        x = self.Relu(x)
        x = self.f5(x)
        return x

    # 权重初始化
    def _initialize_weights(self):
        """
        Xavier权重初始化
        权重初始化公式：上一个节点数 = n,当前权重初始值 = 1/根号n
        """
        for m in self.modules():
            # 判断对象是不是一个实例，惯用方法
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = VGG19()
    # 验证网络 须知输入图像，设定全1矩阵测试 
    input = torch.ones((1, 3, 224, 224))
    output = net(input)
    print(output.shape)

