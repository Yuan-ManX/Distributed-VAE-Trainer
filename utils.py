from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models


class LPIPS(nn.Module):
    """
    LPIPS 类，实现了一个基于学习的感知度量（Learned Perceptual Image Patch Similarity）。

    该类使用预训练的 VGG16 网络作为特征提取器，并在其顶层添加线性层以计算感知相似性。
    通过比较输入图像和目标图像在多个层次的特征差异，度量它们的感知相似性。

    参数:
        use_dropout (bool, optional): 是否在线性层中使用 Dropout，默认为 True。
    """
    def __init__(self, use_dropout=True):
        super().__init__()

        # 定义一个缩放层，用于对输入图像进行归一化处理
        self.scaling_layer = ScalingLayer()
        # 定义 VGG16 网络中各个卷积层的输出通道数
        self.chns = [64, 128, 256, 512, 512]  # vg16 features

        # 加载预训练的 VGG16 模型，并设置其参数不可训练
        self.net = vgg16(pretrained=True, requires_grad=False)

        # 为 VGG16 的每个指定层次添加一个线性层（NetLinLayer），用于计算感知损失
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        # 从预训练文件中加载模型权重
        self.load_from_pretrained()

        # 将所有参数设置为不可训练，确保模型在训练过程中不被更新
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        """
        从预训练文件中加载模型权重。

        参数:
            name (str, optional): 预训练文件的名称，默认为 "vgg_lpips"。
        """
        # 尝试从指定路径加载预训练权重文件 "vgg.pth"
        data = torch.load("vgg.pth", map_location=torch.device("cpu"))

        # 将加载的权重加载到模型中，strict=False 表示允许部分权重不匹配
        self.load_state_dict(
            data,
            strict=False,
        )

    def forward(self, input, target):
        """
        前向传播方法，计算输入图像和目标图像之间的感知相似性。

        参数:
            input (Tensor): 输入图像张量。
            target (Tensor): 目标图像张量。

        返回:
            Tensor: 感知相似性度量值。
        """
        # 对输入图像和目标图像进行缩放归一化处理
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))

        # 通过 VGG16 网络提取特征
        outs0, outs1 = self.net(in0_input), self.net(in1_input)

        # 初始化字典，用于存储特征和特征差异
        feats0, feats1, diffs = {}, {}, {}

        # 获取线性层的列表
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        # 遍历每个特征层次
        for kk in range(len(self.chns)):
            # 对特征进行归一化处理
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            # 计算特征差异的平方
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # 计算感知相似性度量值
        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True) # 对差异应用线性层并计算空间平均值
            for kk in range(len(self.chns))
        ]

        # 将各个层次的相似性度量值相加，得到最终的感知相似性度量值
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    """
    ScalingLayer 类，实现了一个简单的缩放和移位层，用于对输入图像进行归一化处理。

    该层对输入图像的每个通道应用不同的缩放因子和移位因子，以匹配 VGG16 的预处理方式。

    参数:
        无
    """
    def __init__(self):
        super(ScalingLayer, self).__init__()

        # 定义移位因子张量，形状为 (1, 3, 1, 1)，对应于 RGB 三个通道
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )

        # 定义缩放因子张量，形状为 (1, 3, 1, 1)，对应于 RGB 三个通道
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        """
        前向传播方法，应用缩放和移位操作。

        参数:
            inp (Tensor): 输入图像张量。

        返回:
            Tensor: 归一化后的图像张量。
        """
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """
    NetLinLayer 类，实现了一个简单的线性层，通常用于1x1卷积操作。

    该层可以包含一个 Dropout 层（可选），然后是一个1x1卷积层，用于将输入通道数映射到输出通道数。
    通常用于将特征差异映射到感知相似性度量值。

    参数:
        chn_in (int): 输入通道数。
        chn_out (int, optional): 输出通道数，默认为1。
        use_dropout (bool, optional): 是否使用 Dropout 层，默认为 False。
    """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        # 初始化一个空列表，用于存储网络层
        # 如果使用 Dropout，则添加一个 Dropout 层
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )

        # 添加一个1x1卷积层，将输入通道数映射到输出通道数
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]

        # 将层列表转换为 Sequential 模块
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    """
    vgg16 类，实现了一个基于预训练 VGG16 的特征提取器。

    该类加载预训练的 VGG16 模型，并将其特征提取部分拆分为多个切片（slice）。
    每个切片包含 VGG16 网络中连续的若干层，用于提取不同层次的特征。

    参数:
        requires_grad (bool, optional): 是否需要计算梯度，默认为 False。
        pretrained (bool, optional): 是否使用预训练的权重，默认为 True。
    """
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()

        # 加载预训练的 VGG16 模型，并获取其特征提取部分
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features

        # 初始化五个切片，每个切片包含 VGG16 中连续的若干层
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        # 切片数量
        self.N_slices = 5

        # 将前4层添加到第一个切片
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # 将第4到8层添加到第二个切片
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        
        # 将第9到15层添加到第三个切片
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        
        # 将第16到22层添加到第四个切片
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        # 将第23到29层添加到第五个切片
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        # 如果不需要计算梯度，则将所有参数设置为不可训练
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        """
        前向传播方法，应用 VGG16 特征提取器。

        参数:
            X (Tensor): 输入张量

        返回:
            VggOutputs: 一个包含各个切片输出的具名元组
        """
        h = self.slice1(X)       # 应用第一个切片
        h_relu1_2 = h            # 存储第一个切片的输出
        h = self.slice2(h)       # 应用第二个切片
        h_relu2_2 = h            # 存储第二个切片的输出
        h = self.slice3(h)       # 应用第三个切片
        h_relu3_3 = h            # 存储第三个切片的输出
        h = self.slice4(h)       # 应用第四个切片
        h_relu4_3 = h            # 存储第四个切片的输出
        h = self.slice5(h)       # 应用第五个切片
        h_relu5_3 = h            # 存储第五个切片的输出

        # 定义一个具名元组，用于存储各个切片的输出
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )

        # 返回包含各个切片输出的具名元组
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


def normalize_tensor(x, eps=1e-10):
    """
    对输入张量进行归一化处理。

    参数:
        x (Tensor): 输入张量
        eps (float, optional): 一个非常小的常数，避免除零，默认为1e-10

    返回:
        Tensor: 归一化后的张量
    """
    # 计算每个样本在指定维度上的 L2 范数
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    # 对输入张量进行归一化
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    """
    对输入张量在空间维度（高度和宽度）上计算平均值。

    参数:
        x (Tensor): 输入张量
        keepdim (bool, optional): 是否保留维度，默认为 True

    返回:
        Tensor: 空间平均后的张量
    """
    return x.mean([2, 3], keepdim=keepdim)


class PatchDiscriminator(nn.Module):
    """
    PatchDiscriminator 类，实现了一个基于补丁的判别器。

    该判别器通过多个卷积层逐步提取输入图像的特征，并在不同的特征层次上应用二分类器。
    最终将所有二分类器的输出相加，得到判别结果，用于区分真实图像和生成图像。

    参数:
        无
    """
    def __init__(self):
        super(PatchDiscriminator, self).__init__()

        # 定义一个缩放层，用于对输入图像进行归一化处理
        self.scaling_layer = ScalingLayer()

        # 加载预训练的 VGG16 模型
        _vgg = models.vgg16(pretrained=True)

        # 将 VGG16 的特征提取部分拆分为五个切片，每个切片包含连续的若干层
        self.slice1 = nn.Sequential(_vgg.features[:4])        # 第一个切片，包含前4层
        self.slice2 = nn.Sequential(_vgg.features[4:9])       # 第二个切片，包含第4到8层
        self.slice3 = nn.Sequential(_vgg.features[9:16])      # 第三个切片，包含第9到15层
        self.slice4 = nn.Sequential(_vgg.features[16:23])     # 第四个切片，包含第16到22层
        self.slice5 = nn.Sequential(_vgg.features[23:30])     # 第五个切片，包含第23到29层

        # 定义第一个二分类器，应用在第一个切片输出的特征图上
        self.binary_classifier1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),  # 卷积层，输出通道数为32，卷积核大小为4，步幅为4
            nn.ReLU(),  # ReLU 激活函数
            nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),  # 卷积层，输出通道数为1，卷积核大小为4，步幅为4
        )
        # 初始化最后一个卷积层的权重为零
        nn.init.zeros_(self.binary_classifier1[-1].weight)

        # 定义第二个二分类器，应用在第二个切片输出的特征图上
        self.binary_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True), # 卷积层，输出通道数为64，卷积核大小为4，步幅为4
            nn.ReLU(),  # ReLU 激活函数
            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),  # 卷积层，输出通道数为1，卷积核大小为2，步幅为2
        )
        # 初始化最后一个卷积层的权重为零
        nn.init.zeros_(self.binary_classifier2[-1].weight)

        # 定义第三个二分类器，应用在第三个切片输出的特征图上
        self.binary_classifier3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),  # 卷积层，输出通道数为128，卷积核大小为2，步幅为2
            nn.ReLU(),  # ReLU 激活函数
            nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),  # 卷积层，输出通道数为1，卷积核大小为2，步幅为2
        )
        # 初始化最后一个卷积层的权重为零
        nn.init.zeros_(self.binary_classifier3[-1].weight)

        # 定义第四个二分类器，应用在第四个切片输出的特征图上
        self.binary_classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True), # 卷积层，输出通道数为1，卷积核大小为2，步幅为2
        )
        # 初始化最后一个卷积层的权重为零
        nn.init.zeros_(self.binary_classifier4[-1].weight)

        # 定义第五个二分类器，应用在第五个切片输出的特征图上
        self.binary_classifier5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True), # 卷积层，输出通道数为1，卷积核大小为1，步幅为1
        )
        # 初始化最后一个卷积层的权重为零
        nn.init.zeros_(self.binary_classifier5[-1].weight)

    def forward(self, x):
        """
        前向传播方法，应用判别器。

        参数:
            x (Tensor): 输入图像张量

        返回:
            Tensor: 判别结果张量
        """
        # 对输入图像进行缩放归一化处理
        x = self.scaling_layer(x)

        features1 = self.slice1(x)       # 应用第一个切片，提取特征
        features2 = self.slice2(features1)  # 应用第二个切片，提取特征
        features3 = self.slice3(features2)  # 应用第三个切片，提取特征
        features4 = self.slice4(features3)  # 应用第四个切片，提取特征
        features5 = self.slice5(features4)  # 应用第五个切片，提取特征

        # 输出特征图的形状示例
        # torch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32]) torch.Size([1, 512, 16, 16])

        # 应用第一个二分类器，得到判别结果
        bc1 = self.binary_classifier1(features1).flatten(1)
        # 应用第二个二分类器，得到判别结果
        bc2 = self.binary_classifier2(features2).flatten(1)
        # 应用第三个二分类器，得到判别结果
        bc3 = self.binary_classifier3(features3).flatten(1)
        # 应用第四个二分类器，得到判别结果
        bc4 = self.binary_classifier4(features4).flatten(1)
        # 应用第五个二分类器，得到判别结果
        bc5 = self.binary_classifier5(features5).flatten(1)

        # 将所有二分类器的输出相加，得到最终的判别结果
        return bc1 + bc2 + bc3 + bc4 + bc5


# 定义高通和低通滤波器系数
dec_lo, dec_hi = (
    torch.Tensor([-0.1768, 0.3536, 1.0607, 0.3536, -0.1768, 0.0000]),  # 低通滤波器系数
    torch.Tensor([0.0000, -0.0000, 0.3536, -0.7071, 0.3536, -0.0000]),  # 高通滤波器系数
)


# 构建二维滤波器组
filters = torch.stack(
    [
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # 低通-低通滤波器
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # 低通-高通滤波器
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # 高通-低通滤波器
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),  # 高通-高通滤波器
    ],
    dim=0,  # 在第0维上堆叠，形成形状为 (4, 6, 6) 的张量
)


# 为后续的二维卷积操作扩展滤波器的维度
filters_expanded = filters.unsqueeze(1)  # 在第1维上扩展，得到形状为 (4, 1, 6, 6) 的张量


def prepare_filter(device):
    """
    将预定义的滤波器移动到指定的计算设备上。

    参数:
        device (str): 计算设备的名称，如 'cuda' 或 'cpu'
    """
     # 声明使用全局变量 filters_expanded
    global filters_expanded
    # 将滤波器移动到指定的设备
    filters_expanded = filters_expanded.to(device)


def wavelet_transform_multi_channel(x, levels=4):
    """
    对输入的多通道张量应用多通道小波变换。

    该函数对输入张量的每个通道应用二维小波变换，将图像分解为四个子带：
    低频、水平高频、垂直高频和对角高频。

    参数:
        x (Tensor): 输入张量，形状为 (B, C, H, W)
        levels (int, optional): 小波变换的层数，默认为4

    返回:
        Tensor: 变换后的张量，形状为 (B, 4*C, H_out, W_out)
    """
    # 获取输入张量的批次大小、通道数、高度和宽度
    B, C, H, W = x.shape
    # 对输入张量进行填充，填充宽度为2，以确保卷积操作后尺寸一致
    padded = torch.nn.functional.pad(x, (2, 2, 2, 2))

    # 使用预定义的全局滤波器
    global filters_expanded

    # 初始化一个空列表，用于存储每个通道的变换结果
    ress = []

    # 对每个通道应用二维卷积，得到四个子带
    for ch in range(C):
        res = torch.nn.functional.conv2d(
            padded[:, ch : ch + 1], filters_expanded, stride=2
        )
        ress.append(res)  # 将结果添加到列表中
    
    # 将所有通道的变换结果在通道维度上拼接起来
    res = torch.cat(ress, dim=1)

    # 获取变换后张量的高度和宽度
    H_out, W_out = res.shape[2], res.shape[3]

    # 重塑张量形状，从 (B, 4*C, H_out, W_out) 到 (B, C, 4, H_out, W_out)
    res = res.view(B, C, 4, H_out, W_out)

    # 再次重塑张量形状，从 (B, C, 4, H_out, W_out) 到 (B, 4*C, H_out, W_out)
    res = res.view(B, 4 * C, H_out, W_out)

    # 返回变换后的张量
    return res


def test_patch_discriminator():
    """
    测试 PatchDiscriminator 类的函数。

    该函数实例化一个 PatchDiscriminator 对象，将其移动到 CUDA 设备，
    并通过一个随机生成的输入张量进行前向传播，打印输出张量的形状。
    """
    # 实例化 PatchDiscriminator 对象并移动到 CUDA 设备
    vggDiscriminator = PatchDiscriminator().cuda()
    # 生成随机输入张量并通过判别器
    x = vggDiscriminator(torch.randn(1, 3, 256, 256).cuda())
    # 打印输出张量的形状
    print(x.shape)


if __name__ == "__main__":

    # 实例化 PatchDiscriminator 对象并移动到 CUDA 设备
    vggDiscriminator = PatchDiscriminator().cuda()

    # 生成随机输入张量并通过判别器
    x = vggDiscriminator(torch.randn(1, 3, 256, 256).cuda())

    # 打印输出张量的形状
    print(x.shape)
