import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange

from utils import wavelet_transform_multi_channel


def swish(x) -> Tensor:
    """
    Swish 激活函数。

    Swish 是一种自门控激活函数，定义为 x * sigmoid(x)。
    它在许多深度学习任务中表现良好，能够帮助模型更好地捕捉非线性关系。

    参数:
        x (Tensor): 输入张量

    返回:
        Tensor: 应用 Swish 激活后的张量
    """
    return x * torch.sigmoid(x)


# class StandardizedC2d(nn.Conv2d):
#     """
#     StandardizedC2d 类，实现了一个带有权重标准化的二维卷积层。

#     该类继承自 nn.Conv2d，并在每次前向传播时对卷积权重进行标准化处理。
#     通过逐步调整标准化强度，使得权重在训练过程中逐渐稳定。

#     参数:
#         *args: 传递给 nn.Conv2d 的位置参数
#         **kwargs: 传递给 nn.Conv2d 的关键字参数
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.step = 0  # 初始化标准化步数计数器

#     def forward(self, input):
#         output = super().forward(input)
#         # 标准化卷积权重
#         if self.step < 1000:
#             with torch.no_grad():
#                 std = output.std().item()  # 计算输出标准差
#                 normalize_term = (std + 1e-6)**(100/(self.step + 100))  # 计算标准化系数
#                 self.step += 1  # 增加步数计数器
#                 self.weight.data.div_(normalize_term)
#                 self.bias.data.div_(normalize_term)
#                 output.div_(normalize_term)
#                 # 同步权重，广播到所有进程
#                 torch.distributed.broadcast(self.weight.data, 0)
#                 torch.distributed.broadcast(self.bias.data, 0)

#         return output


# 使用标准的二维卷积层
StandardizedC2d = nn.Conv2d


class FP32GroupNorm(nn.GroupNorm):
    """
    FP32GroupNorm 类，实现了一个使用 32 位浮点数的组归一化层。

    该类继承自 nn.GroupNorm，并在前向传播时将输入转换为 32 位浮点数进行处理。
    这有助于在混合精度训练中保持数值稳定性。

    参数:
        *args: 传递给 nn.GroupNorm 的位置参数
        **kwargs: 传递给 nn.GroupNorm 的关键字参数
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """
        前向传播方法，应用组归一化。

        参数:
            input (Tensor): 输入张量

        返回:
            Tensor: 应用组归一化后的张量，类型与输入相同
        """
        # 将输入转换为 32 位浮点数
        output = F.group_norm(
            input.float(),
            self.num_groups,  # 组数
            self.weight.float() if self.weight is not None else None,  # 权重
            self.bias.float() if self.bias is not None else None,  # 偏置
            self.eps,  # 小的常数，避免除零
        )
        # 将输出转换回与输入相同的类型
        return output.type_as(input)


class AttnBlock(nn.Module):
    """
    AttnBlock 类，实现了一个自注意力块。

    该块包括归一化层、标准化卷积层用于生成查询（Q）、键（K）和值（V），以及缩放点积注意力机制。
    通过残差连接，将注意力机制的输出与输入相加。

    参数:
        in_channels (int): 输入通道数
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 存储输入通道数
        self.in_channels = in_channels

        # 每个注意力头的维度
        self.head_dim = 64
        # 计算注意力头的数量
        self.num_heads = in_channels // self.head_dim

        # 定义组归一化层，32 个组，eps 为 1e-6，启用仿射变换
        self.norm = FP32GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        # 定义用于生成 Q、K、V 的标准化卷积层，输出通道数为 in_channels * 3，卷积核大小为 1，不使用偏置
        self.qkv = StandardizedC2d(
            in_channels, in_channels * 3, kernel_size=1, bias=False
        )

        # 定义输出投影的标准化卷积层，输出通道数为 in_channels，卷积核大小为 1，不使用偏置
        self.proj_out = StandardizedC2d(
            in_channels, in_channels, kernel_size=1, bias=False
        )

        # 初始化输出投影层的权重，使用正态分布，标准差为 0.2 / sqrt(in_channels)
        nn.init.normal_(self.proj_out.weight, std=0.2 / math.sqrt(in_channels))

    def attention(self, h_) -> Tensor:
        """
        应用自注意力机制。

        参数:
            h_ (Tensor): 输入特征张量

        返回:
            Tensor: 注意力机制的输出张量
        """
        # 应用归一化
        h_ = self.norm(h_)

        # 生成 Q、K、V
        qkv = self.qkv(h_)

        # 分割为 Q、K、V
        q, k, v = qkv.chunk(3, dim=1)

        # 获取张量形状
        b, c, h, w = q.shape

        # 重排 Q、K、V 的形状以适应多头注意力机制
        q = rearrange(
            q, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        k = rearrange(
            k, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        v = rearrange(
            v, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )

        # 应用缩放点积注意力机制
        h_ = F.scaled_dot_product_attention(q, k, v)

        # 将输出重排回原始形状
        h_ = rearrange(h_, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return h_

    def forward(self, x) -> Tensor:
        """
        前向传播方法，应用自注意力块。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 自注意力块的输出张量
        """
        # 应用注意力机制并添加残差连接
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    ResnetBlock 类，实现了一个残差块（Residual Block）。

    该块结合了归一化、Swish 激活函数和卷积层，并通过跳跃连接（shortcut）实现残差学习。
    如果输入和输出通道数不一致，则使用1x1卷积进行通道数的匹配。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int, optional): 输出通道数。如果为 None，则输出通道数与输入通道数相同。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 存储输入通道数
        self.in_channels = in_channels

        # 如果 out_channels 为 None，则输出通道数与输入通道数相同；否则，使用指定的输出通道数
        out_channels = in_channels if out_channels is None else out_channels

        # 存储输出通道数
        self.out_channels = out_channels

        # 定义第一个归一化层，使用 FP32GroupNorm，32 个组，eps 为 1e-6，启用仿射变换
        self.norm1 = FP32GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        # 定义第一个卷积层，使用 StandardizedC2d，卷积核大小为 3，步幅为 1，填充为 1
        self.conv1 = StandardizedC2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # 定义第二个归一化层，使用 FP32GroupNorm，32 个组，eps 为 1e-6，启用仿射变换
        self.norm2 = FP32GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )

        # 定义第二个卷积层，使用 StandardizedC2d，卷积核大小为 3，步幅为 1，填充为 1
        self.conv2 = StandardizedC2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # 如果输入和输出通道数不一致，则定义1x1卷积层进行通道数的匹配
        if self.in_channels != self.out_channels:
            self.nin_shortcut = StandardizedC2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

        # 初始化第二个卷积层的权重，使用非常小的标准差，并初始化偏置为零
        nn.init.normal_(self.conv2.weight, std=0.0001 / self.out_channels)
        nn.init.zeros_(self.conv2.bias)

        # 初始化一个计数器，用于调试或其他用途
        self.counter = 0

    def forward(self, x):
        """
        前向传播方法，应用残差块。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 残差块的输出张量
        """
        # 如果计数器小于5000，则执行以下操作（用于调试或特定训练策略）
        # 目前被注释掉
        # if self.counter < 5000:
        #     self.counter += 1
        #     h = 0
        # else:

        # 保留输入作为残差
        h = x

        h = self.norm1(h)  # 应用第一个归一化层
        h = swish(h)       # 应用 Swish 激活函数
        h = self.conv1(h)  # 应用第一个卷积层
        h = self.norm2(h)  # 应用第二个归一化层
        h = swish(h)       # 应用 Swish 激活函数
        h = self.conv2(h)  # 应用第二个卷积层

        if self.in_channels != self.out_channels:
            # 如果输入和输出通道数不一致，则应用1x1卷积进行通道数的匹配
            x = self.nin_shortcut(x)
        
        # 添加残差连接
        return x + h


class Downsample(nn.Module):
    """
    Downsample 类，实现了一个下采样模块。

    该模块使用步幅为2的卷积层进行下采样，并通过填充保持输出尺寸的一致性。

    参数:
        in_channels (int): 输入通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()

        # 定义卷积层，使用 StandardizedC2d，卷积核大小为 3，步幅为 2，填充为 0
        self.conv = StandardizedC2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        """
        前向传播方法，应用下采样。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 下采样后的张量
        """
        # 定义填充参数，左右各填充1
        pad = (0, 1, 0, 1)

        # 对输入进行填充，填充值为0
        x = nn.functional.pad(x, pad, mode="constant", value=0)

        # 应用卷积层进行下采样
        x = self.conv(x)

        # 返回下采样后的张量
        return x


class Upsample(nn.Module):
    """
    Upsample 类，实现了一个上采样模块。

    该模块使用最近邻插值进行上采样，然后应用一个卷积层以融合特征。

    参数:
        in_channels (int): 输入通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()

        # 定义卷积层，使用 StandardizedC2d，卷积核大小为 3，步幅为 1，填充为 1
        self.conv = StandardizedC2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """
        前向传播方法，应用上采样。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 上采样后的张量
        """
        # 使用最近邻插值进行上采样
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

        # 应用卷积层
        x = self.conv(x)

        # 返回上采样后的张量
        return x


class Encoder(nn.Module):
    """
    Encoder 类，实现了一个编码器模块。

    该编码器模块通过多层次的残差块和下采样操作，逐步降低输入数据的空间分辨率，同时增加通道数。
    可选地，使用小波变换来预处理输入数据，并在中间层应用注意力机制以捕捉全局依赖关系。

    参数:
        resolution (int): 输入数据的空间分辨率（例如，图像的高度或宽度）。
        in_channels (int): 输入数据的通道数（例如，RGB图像为3）。
        ch (int): 基础通道数，用于确定每个层次的通道数。
        ch_mult (List[int]): 通道数乘数列表，用于逐步增加通道数。每个元素对应一个分辨率层次。
        num_res_blocks (int): 每个分辨率层次中残差块的数量。
        z_channels (int): 编码器输出的通道数。
        use_attn (bool, optional): 是否在中间层使用注意力机制，默认为 True。
        use_wavelet (bool, optional): 是否使用小波变换预处理输入数据，默认为 False。
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        use_attn: bool = True,
        use_wavelet: bool = False,
    ):
        super().__init__()

        self.ch = ch  # 存储基础通道数
        self.num_resolutions = len(ch_mult)  # 计算分辨率层次的数量
        self.num_res_blocks = num_res_blocks  # 存储残差块的数量
        self.resolution = resolution  # 存储输入数据的空间分辨率
        self.in_channels = in_channels  # 存储输入数据的通道数
        self.use_wavelet = use_wavelet  # 存储是否使用小波变换

        if self.use_wavelet:
            # 如果使用小波变换，则定义小波变换函数，并调整第一个卷积层的通道数
            self.wavelet_transform = wavelet_transform_multi_channel
            self.conv_in = StandardizedC2d(
                4 * in_channels, self.ch * 2, kernel_size=3, stride=1, padding=1
            )
            # 调整第一个层次的通道数乘数
            ch_mult[0] *= 2
        else:
            # 如果不使用小波变换，则定义恒等变换，并设置第一个卷积层的通道数
            self.wavelet_transform = nn.Identity()
            self.conv_in = StandardizedC2d(
                in_channels, self.ch, kernel_size=3, stride=1, padding=1
            )

        # 计算每个层次的输入通道数乘数
        curr_res = resolution
        in_ch_mult = (2 if self.use_wavelet else 1,) + tuple(ch_mult)

        # 存储输入通道数乘数
        self.in_ch_mult = in_ch_mult

        # 定义一个 ModuleList 用于存储下采样模块
        self.down = nn.ModuleList()
        # 初始化第一个残差块的输入通道数
        block_in = self.ch

        # 遍历每个分辨率层次
        for i_level in range(self.num_resolutions):
            # 定义一个 ModuleList 用于存储当前层次的残差块
            block = nn.ModuleList()
            # 定义一个 ModuleList 用于存储当前层次的注意力模块
            attn = nn.ModuleList()

            # 计算当前层次的输入通道数
            block_in = ch * in_ch_mult[i_level]
            # 计算当前层次的输出通道数
            block_out = ch * ch_mult[i_level]

            # 在当前层次中添加多个残差块
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out

            # 定义一个 Module 用于存储当前层次的下采样信息
            down = nn.Module()
            # 存储残差块列表
            down.block = block
            # 存储注意力模块列表
            down.attn = attn

            # 如果不是最后一个层次，且不是第一个层次且使用小波变换，则添加下采样模块
            if i_level != self.num_resolutions - 1 and not (
                self.use_wavelet and i_level == 0
            ):
                # 添加下采样模块
                down.downsample = Downsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res // 2
            # 将当前层次的下采样信息添加到 ModuleList 中
            self.down.append(down)

        # 定义中间层
        self.mid = nn.Module()
        # 第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 注意力模块，如果使用注意力机制
        self.mid.attn_1 = AttnBlock(block_in) if use_attn else nn.Identity()

        # 第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 定义输出归一化层和卷积层
        self.norm_out = FP32GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = StandardizedC2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

        # 初始化所有标准化卷积层的偏置为零，所有组归一化层的偏置为零
        for module in self.modules():
            if isinstance(module, StandardizedC2d):
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.GroupNorm):
                nn.init.zeros_(module.bias)

    def forward(self, x) -> Tensor:
        """
        前向传播方法，应用编码器模块。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 编码器输出张量
        """
        # 应用小波变换预处理输入数据
        h = self.wavelet_transform(x)
        # 应用第一个卷积层
        h = self.conv_in(h)

        # 遍历每个分辨率层次
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # 应用残差块
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    # 应用注意力模块（如果有）
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1 and not (
                self.use_wavelet and i_level == 0
            ):
                # 应用下采样模块
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)  # 应用第一个中间残差块
        h = self.mid.attn_1(h)   # 应用中间注意力模块（如果有）
        h = self.mid.block_2(h)  # 应用第二个中间残差块
        h = self.norm_out(h)     # 应用输出归一化
        h = swish(h)             # 应用 Swish 激活函数
        h = self.conv_out(h)     # 应用输出卷积层

        # 返回编码器输出
        return h


class Decoder(nn.Module):
    """
    Decoder 类，实现了一个解码器模块。

    该解码器模块通过多层次的上采样和残差块，逐步恢复输入数据的空间分辨率，同时减少通道数。
    可选地，在中间层应用注意力机制以捕捉全局依赖关系。

    参数:
        ch (int): 基础通道数，用于确定每个层次的通道数。
        out_ch (int): 输出数据的通道数（例如，RGB图像为3）。
        ch_mult (List[int]): 通道数乘数列表，用于逐步减少通道数。每个元素对应一个分辨率层次。
        num_res_blocks (int): 每个分辨率层次中残差块的数量。
        in_channels (int): 输入数据的通道数（通常是编码器输出的通道数）。
        resolution (int): 输出数据的空间分辨率（例如，图像的高度或宽度）。
        z_channels (int): 输入数据的通道数（通常是编码器的输出通道数）。
        use_attn (bool, optional): 是否在中间层使用注意力机制，默认为 True。
    """
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        use_attn: bool = True,
    ):
        super().__init__()

        # 存储基础通道数
        self.ch = ch
        # 计算分辨率层次的数量
        self.num_resolutions = len(ch_mult)
        # 存储残差块的数量
        self.num_res_blocks = num_res_blocks
        # 存储输出数据的空间分辨率
        self.resolution = resolution
        # 存储输入数据的通道数
        self.in_channels = in_channels
        # 计算上采样因子
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # 计算第一个上采样块的输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # 计算第一个上采样块的空间分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # 定义输入数据的空间形状
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # 定义输入卷积层，将输入数据映射到基础通道数
        self.conv_in = StandardizedC2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # 定义中间层
        self.mid = nn.Module()
        # 第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 注意力模块，如果使用注意力机制
        self.mid.attn_1 = AttnBlock(block_in) if use_attn else nn.Identity()
        # 第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 定义上采样模块列表
        self.up = nn.ModuleList()
        # 逆序遍历每个分辨率层次，以实现上采样
        for i_level in reversed(range(self.num_resolutions)):
            # 定义一个 ModuleList 用于存储当前层次的上采样块
            block = nn.ModuleList()
            # 定义一个 ModuleList 用于存储当前层次的注意力模块
            attn = nn.ModuleList()

            # 计算当前层次的上采样输出通道数
            block_out = ch * ch_mult[i_level]

            # 在当前层次中添加多个残差块（num_res_blocks + 1）
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out

            # 定义一个 Module 用于存储当前层次的上采样信息
            up = nn.Module()
            # 存储残差块列表
            up.block = block
            # 存储注意力模块列表
            up.attn = attn

            # 如果不是第一个层次，则添加上采样模块
            if i_level != 0:
                # 添加上采样模块
                up.upsample = Upsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res * 2
            # 将当前层次的上采样信息插入到列表的开头，以实现逆序遍历
            self.up.insert(0, up)

        # 定义输出归一化层和输出卷积层
        self.norm_out = FP32GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = StandardizedC2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

        # 初始化所有标准化卷积层的偏置为零，所有组归一化层的偏置为零
        for module in self.modules():
            if isinstance(module, StandardizedC2d):
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.GroupNorm):
                nn.init.zeros_(module.bias)

    def forward(self, z) -> Tensor:
        """
        前向传播方法，应用解码器模块。

        参数:
            z (Tensor): 输入张量

        返回:
            Tensor: 解码器输出张量
        """
        h = self.conv_in(z)  # 应用输入卷积层
        h = self.mid.block_1(h)  # 应用第一个中间残差块
        h = self.mid.attn_1(h)   # 应用中间注意力模块（如果有）
        h = self.mid.block_2(h)  # 应用第二个中间残差块

        # 逆序遍历每个分辨率层次，以实现上采样
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # 应用残差块
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    # 应用注意力模块（如果有）
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # 应用上采样模块
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)  # 应用输出归一化
        h = swish(h)          # 应用 Swish 激活函数
        h = self.conv_out(h)  # 应用输出卷积层

        # 返回解码器输出
        return h


class DiagonalGaussian(nn.Module):
    """
    DiagonalGaussian 类，实现了一个对角高斯分布采样模块。

    该模块根据输入的均值生成样本。如果 `sample` 参数为 True，则从对角高斯分布中采样；否则，仅返回均值。

    参数:
        sample (bool, optional): 是否进行采样，默认为 True。
        chunk_dim (int, optional): 分割维度，用于分割输入张量，默认为1。
    """
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()

        # 存储是否进行采样
        self.sample = sample
        # 存储分割维度
        self.chunk_dim = chunk_dim

    def forward(self, z) -> Tensor:
        """
        前向传播方法，执行采样或返回均值。

        参数:
            z (Tensor): 输入张量，代表均值

        返回:
            Tensor: 采样结果或均值
        """
        # 获取均值
        mean = z

        if self.sample:
            # 设置标准差为0.00
            std = 0.00
            # 返回均值乘以 (1 + 标准差 * 标准正态分布噪声)
            return mean * (1 + std * torch.randn_like(mean))
        
        else:
            # 返回均值
            return mean


class VAE(nn.Module):
    """
    VAE 类，实现了一个变分自编码器（Variational Autoencoder）。

    该 VAE 模型包含一个编码器（Encoder）和一个解码器（Decoder），用于将输入数据编码为潜在空间表示，
    然后再从潜在空间重建原始数据。潜在空间表示遵循对角高斯分布。

    参数:
        resolution (int): 输入数据的空间分辨率（例如，图像的高度或宽度）。
        in_channels (int): 输入数据的通道数（例如，RGB图像为3）。
        ch (int): 基础通道数，用于确定编码器和解码器中每个层次的通道数。
        out_ch (int): 输出数据的通道数（例如，RGB图像为3）。
        ch_mult (List[int]): 通道数乘数列表，用于逐步增加或减少通道数。每个元素对应一个分辨率层次。
        num_res_blocks (int): 每个分辨率层次中残差块的数量。
        z_channels (int): 潜在空间的通道数，即编码器输出的通道数。
        use_attn (bool): 是否在编码器和解码器中使用注意力机制。
        decoder_also_perform_hr (bool): 是否在解码器中执行高分辨率处理。如果为 True，则在解码器通道数乘数列表中添加一个额外的 4。
        use_wavelet (bool): 是否在编码器中使用小波变换预处理输入数据。
    """
    def __init__(
        self,
        resolution,
        in_channels,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        z_channels,
        use_attn,
        decoder_also_perform_hr,
        use_wavelet,
    ):
        super().__init__()

        # 初始化编码器模块
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            use_attn=use_attn,
            use_wavelet=use_wavelet,
        )

        # 初始化解码器模块
        # 如果 decoder_also_perform_hr 为 True，则在通道数乘数列表中添加一个额外的 4
        self.decoder = Decoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult + [4] if decoder_also_perform_hr else ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            use_attn=use_attn,
        )

        # 初始化对角高斯分布采样模块
        self.reg = DiagonalGaussian()

    def forward(self, x) -> Tensor:
        """
        前向传播方法，应用 VAE 模型。

        参数:
            x (Tensor): 输入张量

        返回:
            Tuple[Tensor, Tensor]: 解码器输出张量和潜在空间表示张量
        """
        # 通过编码器将输入数据编码为潜在空间表示
        z = self.encoder(x)
        # 对潜在空间表示进行对角高斯分布采样
        z_s = self.reg(z)
        # 通过解码器从采样后的潜在空间表示重建数据
        decz = self.decoder(z_s)
        # 返回重建的数据和潜在空间表示
        return decz, z


if __name__ == "__main__":
    
    from utils import prepare_filter

    prepare_filter("cuda")

    # 初始化 VAE 模型，设置相关参数
    vae = VAE(
        resolution=256,            # 输入数据的空间分辨率，设置为256
        in_channels=3,             # 输入数据的通道数，设置为3（RGB图像）
        ch=64,                     # 基础通道数，设置为64
        out_ch=3,                  # 输出数据的通道数，设置为3（RGB图像）
        ch_mult=[1, 2, 4, 4, 4],   # 通道数乘数列表，设置为 [1, 2, 4, 4, 4]
        num_res_blocks=2,          # 每个分辨率层次中残差块的数量，设置为2
        z_channels=16 * 4,         # 潜在空间的通道数，设置为64
        use_attn=False,            # 不使用注意力机制
        decoder_also_perform_hr=False,  # 不在解码器中执行高分辨率处理
        use_wavelet=False,         # 不使用小波变换
    )
    vae.eval().to("cuda")
    # 生成一个随机输入张量，形状为 (1, 3, 256, 256)，并移动到 CUDA 设备
    x = torch.randn(1, 3, 256, 256).to("cuda")

    # 前向传播，获取重建数据和潜在空间表示
    decz, z = vae(x)
    # 打印重建数据和潜在空间表示的形状
    print(decz.shape, z.shape)

    # 以下是一些被注释掉的代码，用于模型权重初始化和保存
    # 如果需要使用，可以取消注释并根据实际情况进行调整

    # from unit_activation_reinitializer import adjust_weight_init
    # from torchvision import transforms
    # import torchvision

    # 初始化 CIFAR-10 数据集，进行预处理
    # train_dataset = torchvision.datasets.CIFAR10(
    #     root="./data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize((256, 256)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]
    #     ),
    # )

    # 调整权重初始化参数
    # initial_std, layer_weight_std = adjust_weight_init(
    #     vae,
    #     dataset=train_dataset,
    #     device="cuda:0",
    #     batch_size=64,
    #     num_workers=0,
    #     tol=0.1,
    #     max_iters=10,
    #     exclude_layers=[FP32GroupNorm, nn.LayerNorm],
    # )
    
    # 保存权重初始化参数
    # # save initial_std and layer_weight_std
    # torch.save(initial_std, "initial_std.pth")
    # torch.save(layer_weight_std, "layer_weight_std.pth")

    # 打印调整后的权重标准差
    # print("\nAdjusted Weight Standard Deviations. Before -> After:")
    # for layer_name, std in layer_weight_std.items():
    #     print(
    #         f"Layer {layer_name}, Changed STD from \n   {initial_std[layer_name]:.4f} -> STD {std:.4f}\n"
    #     )

    # 打印调整后的权重标准差字典
    # print(layer_weight_std)
