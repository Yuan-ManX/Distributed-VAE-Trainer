import logging
import os
import random
import click
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import GaussianBlur
from transformers import get_cosine_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import wandb
from vae import VAE
from utils import LPIPS, PatchDiscriminator, prepare_filter
import time


class GradNormFunction(torch.autograd.Function):
    """
    GradNormFunction 类，实现了一个自定义的梯度归一化操作。

    该函数在前向传播时返回输入张量的副本，在反向传播时对梯度进行归一化处理。
    归一化过程包括计算梯度的范数，并根据给定的权重对其进行缩放。
    """
    @staticmethod
    def forward(ctx, x, weight):
        """
        前向传播方法。

        参数:
            ctx: 上下文对象，用于在反向传播时传递信息。
            x (Tensor): 输入张量。
            weight (Tensor): 用于梯度归一化的权重张量。

        返回:
            Tensor: 输入张量的副本。
        """
        # 将权重保存到上下文中，以便在反向传播时使用
        ctx.save_for_backward(weight)
        # 返回输入张量的副本
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播方法，对梯度进行归一化处理。

        参数:
            ctx: 上下文对象，包含前向传播时保存的权重。
            grad_output (Tensor): 从后续层传递回来的梯度。

        返回:
            Tuple[Tensor, None]: 归一化后的梯度张量和 None（因为输入张量没有梯度）。
        """
        # 从上下文中获取保存的权重
        weight = ctx.saved_tensors[0]

        # 计算梯度输出的 L2 范数（跨所有维度），然后计算所有样本的平均值
        # grad_output_norm = torch.linalg.vector_norm(
        #     grad_output, dim=list(range(1, len(grad_output.shape))), keepdim=True
        # ).mean()

        # 计算梯度的 L2 范数并取平均值
        grad_output_norm = torch.norm(grad_output).mean().item()

        # 在所有节点上对梯度范数进行平均（假设使用分布式训练）
        # nccl over all nodes
        grad_output_norm = avg_scalar_over_nodes(
            grad_output_norm, device=grad_output.device
        )

        # 对梯度进行归一化处理：grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)
        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        # 返回归一化后的梯度张量和 None（因为输入张量没有梯度）
        return grad_output_normalized, None


def gradnorm(x, weight=1.0):
    """
    对输入张量应用梯度归一化。

    参数:
        x (Tensor): 输入张量。
        weight (float, optional): 用于梯度归一化的权重，默认为1.0。

    返回:
        Tensor: 应用梯度归一化后的张量。
    """
    # 将权重转换为与输入张量相同设备的张量
    weight = torch.tensor(weight, device=x.device)
    # 应用自定义的梯度归一化函数
    return GradNormFunction.apply(x, weight)


# 禁用梯度计算，节省内存
@torch.no_grad()
def avg_scalar_over_nodes(value: float, device):
    """
    在所有节点上对标量值进行平均。

    该函数用于分布式训练中，将每个节点的标量值进行平均。

    参数:
        value (float): 要平均的标量值。
        device (torch.device): 计算设备。

    返回:
        float: 平均后的标量值。
    """
    # 将标量值转换为张量，并移动到指定设备
    value = torch.tensor(value, device=device)
    # 在所有节点上执行平均操作
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    # 返回标量值
    return value.item()


def gan_disc_loss(real_preds, fake_preds, disc_type="bce"):
    """
    计算生成对抗网络（GAN）判别器的损失。

    该函数根据判别器类型计算真实样本和生成样本的损失，并返回平均损失、真实预测的平均值、生成预测的平均值以及准确率。

    参数:
        real_preds (Tensor): 判别器对真实样本的预测输出。
        fake_preds (Tensor): 判别器对生成样本的预测输出。
        disc_type (str, optional): 判别器类型，默认为 'bce'（二元交叉熵）。可选值为 'hinge'。

    返回:
        Tuple[Tensor, float, float, float]: 平均损失、真实预测的平均值、生成预测的平均值以及准确率。
    """
    if disc_type == "bce":
        # 计算真实样本的二元交叉熵损失
        real_loss = nn.functional.binary_cross_entropy_with_logits(
            real_preds, torch.ones_like(real_preds)
        )
        # 计算生成样本的二元交叉熵损失
        fake_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_preds, torch.zeros_like(fake_preds)
        )
        # 计算真实预测的平均值
        avg_real_preds = real_preds.mean().item()
        # 计算生成预测的平均值
        avg_fake_preds = fake_preds.mean().item()

        # 计算准确率
        with torch.no_grad():  # 禁用梯度计算
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

    if disc_type == "hinge":
        # 计算真实样本的 Hinge 损失
        real_loss = nn.functional.relu(1 - real_preds).mean()
        # 计算生成样本的 Hinge 损失
        fake_loss = nn.functional.relu(1 + fake_preds).mean()

        # 计算准确率
        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

        # 计算真实预测的平均值
        avg_real_preds = real_preds.mean().item()
        # 计算生成预测的平均值
        avg_fake_preds = fake_preds.mean().item()

    # 返回平均损失、真实预测的平均值、生成预测的平均值以及准确率
    return (real_loss + fake_loss) * 0.5, avg_real_preds, avg_fake_preds, acc


MAX_WIDTH = 512

# 定义一个标准的图像变换组合
this_transform = transforms.Compose(
    [
        transforms.ToTensor(), # 将 PIL 图像或 NumPy 数组转换为张量，并归一化到 [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 标准化张量，使其均值为0，标准差为1
        transforms.CenterCrop(512), # 中心裁剪图像到 512x512
        transforms.Resize(MAX_WIDTH), # 调整图像大小到 MAX_WIDTH（512）
    ]
)


def this_transform_random_crop_resize(x, width=MAX_WIDTH):
    """
    对输入图像应用随机裁剪或调整大小。

    该函数首先将输入转换为张量并标准化，然后以 50% 的概率随机选择裁剪或调整大小。
    如果选择裁剪，则随机裁剪图像到指定宽度；否则，先调整大小到指定宽度，再随机裁剪。

    参数:
        x: 输入图像，可以是 PIL 图像、NumPy 数组或张量
        width (int, optional): 裁剪或调整大小的目标宽度，默认为 MAX_WIDTH（512）

    返回:
        Tensor: 变换后的图像张量
    """
    # 将输入转换为张量，并归一化到 [0, 1]
    x = transforms.ToTensor()(x)
    # 标准化张量，使其均值为0，标准差为1
    x = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)

    if random.random() < 0.5:
        # 如果随机数小于0.5，则进行随机裁剪
        x = transforms.RandomCrop(width)(x)
    else:
        # 否则，先调整大小到指定宽度，再进行随机裁剪
        x = transforms.Resize(width)(x)
        x = transforms.RandomCrop(width)(x)

    # 返回变换后的图像张量
    return x


def create_dataloader(url, batch_size, num_workers, do_shuffle=True, just_resize=False):
    """
    创建一个数据加载器，用于从指定的 URL 加载数据。

    该函数使用 WebDataset 从指定的 URL 加载数据，并应用相应的图像变换。
    如果 `just_resize` 为 True，则仅调整图像大小；否则，应用随机裁剪或调整大小。

    参数:
        url (str): 数据集的 URL
        batch_size (int): 每个批次的样本数量
        num_workers (int): 用于数据加载的子进程数量
        do_shuffle (bool, optional): 是否打乱数据，默认为 True
        just_resize (bool, optional): 是否仅调整图像大小，默认为 False

    返回:
        WebLoader: 创建的数据加载器
    """
    # 创建 WebDataset 对象，指定数据集的 URL，并使用节点分割器和工作者分割器
    dataset = wds.WebDataset(
        url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker
    )
    # 如果需要打乱数据，则对数据集进行打乱操作
    dataset = dataset.shuffle(1000) if do_shuffle else dataset

    # 解码图像为 RGB 格式，并将图像和标签组合成元组
    dataset = (
        dataset.decode("rgb")
        .to_tuple("jpg;png")
        .map_tuple(
            this_transform_random_crop_resize if not just_resize else this_transform  # 根据参数选择变换
        )
    )

    # 创建 WebLoader 对象，加载数据集
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size, # 设置批次大小
        shuffle=False, # 在 WebLoader 中不进行打乱，因为已经在 WebDataset 中打乱
        num_workers=num_workers, # 设置子进程数量
        pin_memory=True, # 启用内存固定，加速数据传输
    )
    # 返回创建的数据加载器
    return loader


def blurriness_heatmap(input_image):
    """
    生成输入图像的模糊度热图。

    该函数通过计算图像的边缘响应，并将其转换为模糊度度量，生成模糊度热图。

    参数:
        input_image (Tensor): 输入图像张量，形状为 (B, C, H, W)

    返回:
        Tensor: 模糊度热图张量，形状为 (B, 3, H, W)
    """
    # 将输入图像转换为灰度图像
    # 在通道维度上求均值，得到形状为 (B, 1, H, W)
    grayscale_image = input_image.mean(dim=1, keepdim=True)

    # 定义拉普拉斯卷积核，用于检测边缘
    laplacian_kernel = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, -20, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    # 重塑为 (1, 1, 5, 5) 的张量
    laplacian_kernel = laplacian_kernel.view(1, 1, 5, 5)

    # 将卷积核移动到与输入图像相同的设备
    laplacian_kernel = laplacian_kernel.to(input_image.device)

    # 计算边缘响应
    # 应用卷积，填充宽度为2
    edge_response = F.conv2d(grayscale_image, laplacian_kernel, padding=2)

    # 对边缘响应应用高斯模糊，以平滑边缘
    # 应用高斯模糊
    edge_magnitude = GaussianBlur(kernel_size=(13, 13), sigma=(2.0, 2.0))(
        edge_response.abs()
    )

    # 将边缘幅度归一化到 [0, 1] 范围
    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (
        edge_magnitude.max() - edge_magnitude.min() + 1e-8
    )

    # 计算模糊度映射，模糊度越高，值越接近1
    blurriness_map = 1 - edge_magnitude

    # 将模糊度映射中小于0.8的部分设为0，其余部分保持不变
    blurriness_map = torch.where(
        blurriness_map < 0.8, torch.zeros_like(blurriness_map), blurriness_map
    )

    # 将模糊度映射在通道维度上重复3次，以匹配RGB图像的通道数
    return blurriness_map.repeat(1, 3, 1, 1)  # 返回形状为 (B, 3, H, W) 的模糊度热图


def vae_loss_function(x, x_reconstructed, z, do_pool=True, do_recon=False):
    """
    计算变分自编码器（VAE）的损失函数。

    该函数计算重建损失（reconstruction loss）和潜在空间损失（latent space loss），并返回总损失。
    可选的参数允许选择是否进行下采样以及是否计算重建损失。

    参数:
        x (Tensor): 原始输入图像张量，形状为 (B, C, H, W)
        x_reconstructed (Tensor): 重建后的图像张量，形状为 (B, C, H, W)
        z (Tensor): 潜在空间表示张量，形状为 (B, Z_dim)
        do_pool (bool, optional): 是否对图像进行下采样，默认为 True
        do_recon (bool, optional): 是否计算重建损失，默认为 False

    返回:
        Tuple[Tensor, Dict[str, float]]: 总损失和包含各项损失的字典
    """
    # 如果需要计算重建损失
    if do_recon:
        if do_pool:
            # 对重建图像和原始图像进行下采样，下采样因子为1/16（即缩小16倍）
            x_reconstructed_down = F.interpolate(
                x_reconstructed, scale_factor=1 / 16, mode="area"
            )
            x_down = F.interpolate(x, scale_factor=1 / 16, mode="area")
            # 计算重建损失，使用 L1 损失（绝对误差均值）
            recon_loss = ((x_reconstructed_down - x_down)).abs().mean()
        else:
            # 如果不需要下采样，直接使用原始图像和重建图像计算重建损失
            x_reconstructed_down = x_reconstructed
            x_down = x

            # 使用模糊度热图对重建损失进行加权
            recon_loss = (
                ((x_reconstructed_down - x_down) * blurriness_heatmap(x_down))
                .abs()
                .mean()
            )
            # 获取重建损失的标量值
            recon_loss_item = recon_loss.item()
    else:
        # 如果不需要计算重建损失，则将重建损失设为0
        recon_loss = 0
        recon_loss_item = 0

    # 计算潜在空间损失的逐元素均值
    elewise_mean_loss = z.pow(2)
    zloss = elewise_mean_loss.mean()

    # 在不计算梯度的情况下，计算实际的均值损失和 KL 散度损失
    with torch.no_grad():
        actual_mean_loss = elewise_mean_loss.mean()
        actual_ks_loss = actual_mean_loss.mean()

    # 计算总损失，重建损失的权重为0，潜在空间损失的权重为0.1
    vae_loss = recon_loss * 0.0 + zloss * 0.1

    # 返回总损失和包含各项损失的字典
    return vae_loss, {
        "recon_loss": recon_loss_item,  # 重建损失的标量值
        "kl_loss": actual_ks_loss.item(),  # KL 散度损失的标量值
        "average_of_abs_z": z.abs().mean().item(),  # 潜在空间表示绝对值的均值
        "std_of_abs_z": z.abs().std().item(),  # 潜在空间表示绝对值的标准差
        "average_of_logvar": 0.0,  # 对数方差均值的占位符
        "std_of_logvar": 0.0,  # 对数方差标准差的占位符
    }


def cleanup():
    """
    清理分布式训练环境。

    该函数销毁进程组，释放分布式训练相关的资源。
    """
    # 销毁进程组，清理分布式训练环境
    dist.destroy_process_group()


@click.command()
@click.option(
    "--dataset_url", type=str, default="", help="URL for the training dataset"
)
@click.option(
    "--test_dataset_url", type=str, default="", help="URL for the test dataset"
)
@click.option("--num_epochs", type=int, default=2, help="Number of training epochs")
@click.option("--batch_size", type=int, default=8, help="Batch size for training")
@click.option("--do_ganloss", is_flag=True, help="Whether to use GAN loss")
@click.option(
    "--learning_rate_vae", type=float, default=1e-5, help="Learning rate for VAE"
)
@click.option(
    "--learning_rate_disc",
    type=float,
    default=2e-4,
    help="Learning rate for discriminator",
)
@click.option("--vae_resolution", type=int, default=256, help="Resolution for VAE")
@click.option("--vae_in_channels", type=int, default=3, help="Input channels for VAE")
@click.option("--vae_ch", type=int, default=256, help="Base channel size for VAE")
@click.option(
    "--vae_ch_mult", type=str, default="1,2,4,4", help="Channel multipliers for VAE"
)
@click.option(
    "--vae_num_res_blocks",
    type=int,
    default=2,
    help="Number of residual blocks for VAE",
)
@click.option(
    "--vae_z_channels", type=int, default=16, help="Number of latent channels for VAE"
)
@click.option("--run_name", type=str, default="run", help="Name of the run for wandb")
@click.option(
    "--max_steps", type=int, default=1000, help="Maximum number of steps to train for"
)
@click.option(
    "--evaluate_every_n_steps", type=int, default=250, help="Evaluate every n steps"
)
@click.option("--load_path", type=str, default=None, help="Path to load the model from")
@click.option("--do_clamp", is_flag=True, help="Whether to clamp the latent codes")
@click.option(
    "--clamp_th", type=float, default=8.0, help="Clamp threshold for the latent codes"
)
@click.option(
    "--max_spatial_dim",
    type=int,
    default=256,
    help="Maximum spatial dimension for overall training",
)
@click.option(
    "--do_attn", type=bool, default=False, help="Whether to use attention in the VAE"
)
@click.option(
    "--decoder_also_perform_hr",
    type=bool,
    default=False,
    help="Whether to perform HR decoding in the decoder",
)
@click.option(
    "--project_name",
    type=str,
    default="vae_sweep_attn_lr_width",
    help="Project name for wandb",
)
@click.option(
    "--crop_invariance",
    type=bool,
    default=False,
    help="Whether to perform crop invariance",
)
@click.option(
    "--flip_invariance",
    type=bool,
    default=False,
    help="Whether to perform flip invariance",
)
@click.option(
    "--do_compile",
    type=bool,
    default=False,
    help="Whether to compile the model",
)
@click.option(
    "--use_wavelet",
    type=bool,
    default=False,
    help="Whether to use wavelet transform in the encoder",
)
@click.option(
    "--augment_before_perceptual_loss",
    type=bool,
    default=False,
    help="Whether to augment the images before the perceptual loss",
)
@click.option(
    "--downscale_factor",
    type=int,
    default=16,
    help="Downscale factor for the latent space",
)
@click.option(
    "--use_lecam",
    type=bool,
    default=False,
    help="Whether to use Lecam",
)
@click.option(
    "--disc_type",
    type=str,
    default="bce",
    help="Discriminator type",
)


def train_ddp(
    dataset_url,
    test_dataset_url,
    num_epochs,
    batch_size,
    do_ganloss,
    learning_rate_vae,
    learning_rate_disc,
    vae_resolution,
    vae_in_channels,
    vae_ch,
    vae_ch_mult,
    vae_num_res_blocks,
    vae_z_channels,
    run_name,
    max_steps,
    evaluate_every_n_steps,
    load_path,
    do_clamp,
    clamp_th,
    max_spatial_dim,
    do_attn,
    decoder_also_perform_hr,
    project_name,
    crop_invariance,
    flip_invariance,
    do_compile,
    use_wavelet,
    augment_before_perceptual_loss,
    downscale_factor,
    use_lecam,
    disc_type,
):
    """
    训练函数，使用分布式数据并行（DDP）进行模型训练。

    参数:
        dataset_url (str): 训练数据集的URL。
        test_dataset_url (str): 测试数据集的URL。
        num_epochs (int): 训练的轮数。
        batch_size (int): 每个批次的大小。
        do_ganloss (bool): 是否使用生成对抗网络损失。
        learning_rate_vae (float): VAE的学习率。
        learning_rate_disc (float): 判别器的学习率。
        vae_resolution (int): VAE的分辨率。
        vae_in_channels (int): VAE输入通道数。
        vae_ch (int): VAE基础通道数。
        vae_ch_mult (str): VAE通道数倍数（以逗号分隔的字符串）。
        vae_num_res_blocks (int): VAE残差块的数量。
        vae_z_channels (int): VAE隐空间通道数。
        run_name (str): 运行名称，用于记录和标识。
        max_steps (int): 最大训练步数。
        evaluate_every_n_steps (int): 每隔多少步进行一次评估。
        load_path (str): 模型加载路径（如果需要加载预训练模型）。
        do_clamp (bool): 是否进行梯度裁剪。
        clamp_th (float): 梯度裁剪的阈值。
        max_spatial_dim (int): 最大空间维度。
        do_attn (bool): 是否使用注意力机制。
        decoder_also_perform_hr (bool): 解码器是否也执行高分辨率操作。
        project_name (str): Weights & Biases项目名称。
        crop_invariance (bool): 裁剪不变性（是否启用）。
        flip_invariance (bool): 翻转不变性（是否启用）。
        do_compile (bool): 是否进行模型编译以加速。
        use_wavelet (bool): 是否使用小波变换。
        augment_before_perceptual_loss (bool): 在感知损失前是否进行数据增强。
        downscale_factor (float): 下采样因子。
        use_lecam (bool): 是否使用LECAM（局部能量约束对抗性方法）。
        disc_type (str): 判别器类型。
    """

    # 设置随机种子以确保结果的可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # 定义训练和测试数据集的索引范围
    start_train = 0
    # 假设每个tar文件包含128个样本，16个tar文件用于训练
    end_train = 128 * 16

    start_test = end_train + 1
    end_test = start_test + 8  # 8个tar文件用于测试

    # 根据索引范围构建训练和测试数据集的URL
    dataset_url = f"/flux_ipadapter_trainer/dataset/art_webdataset/{{{start_train:05d}..{end_train:05d}}}.tar"
    test_dataset_url = f"/flux_ipadapter_trainer/dataset/art_webdataset/{{{start_test:05d}..{end_test:05d}}}.tar"

    # 检查CUDA是否可用，因为DDP需要GPU支持
    assert torch.cuda.is_available(), "CUDA is required for DDP"

    # 初始化分布式进程组
    dist.init_process_group(backend="nccl")

    # 获取当前进程的排名和本地排名，以及总进程数
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 设置当前设备
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)

    # 主进程标识
    master_process = ddp_rank == 0
    print(f"using device: {device}")

    # 如果是主进程，则初始化Weights & Biases（wandb）以进行监控和记录
    if master_process:
        wandb.init(
            project=project_name,
            entity="simo",
            name=run_name,
            config={
                "learning_rate_vae": learning_rate_vae,
                "learning_rate_disc": learning_rate_disc,
                "vae_ch": vae_ch,
                "vae_resolution": vae_resolution,
                "vae_in_channels": vae_in_channels,
                "vae_ch_mult": vae_ch_mult,
                "vae_num_res_blocks": vae_num_res_blocks,
                "vae_z_channels": vae_z_channels,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "do_ganloss": do_ganloss,
                "do_attn": do_attn,
                "use_wavelet": use_wavelet,
            },
        )

    # 初始化VAE模型，并将其移动到GPU
    vae = VAE(
        resolution=vae_resolution,
        in_channels=vae_in_channels,
        ch=vae_ch,
        out_ch=vae_in_channels,
        ch_mult=[int(x) for x in vae_ch_mult.split(",")],
        num_res_blocks=vae_num_res_blocks,
        z_channels=vae_z_channels,
        use_attn=do_attn,
        decoder_also_perform_hr=decoder_also_perform_hr,
        use_wavelet=use_wavelet,
    ).cuda()

    # 初始化判别器模型，并将其移动到GPU
    discriminator = PatchDiscriminator().cuda()
    discriminator.requires_grad_(True)

    # 使用DDP包装VAE模型以支持分布式训练
    vae = DDP(vae, device_ids=[ddp_rank])

    # 准备过滤器（具体实现未提供，假设用于数据预处理）
    prepare_filter(device)

    # 如果需要进行编译优化
    if do_compile:
        # 使用torch.compile编译VAE的编码器，设置fullgraph=False，模式为"max-autotune"
        vae.module.encoder = torch.compile(
            vae.module.encoder, fullgraph=False, mode="max-autotune"
        )
        # 使用torch.compile编译VAE的解码器，设置fullgraph=False，模式为"max-autotune"
        vae.module.decoder = torch.compile(
            vae.module.decoder, fullgraph=False, mode="max-autotune"
        )

    # 使用分布式数据并行（DDP）包装判别器，指定设备ID为ddp_rank
    discriminator = DDP(discriminator, device_ids=[ddp_rank])

    # 设置混合精度上下文，使用cuda设备，精度为bfloat16
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # 定义生成器（VAE）的优化器，使用AdamW算法
    optimizer_G = optim.AdamW(
        [
            # 对于VAE中不包含"conv_in"的参数，学习率为learning_rate_vae / vae_ch
            {
                "params": [p for n, p in vae.named_parameters() if "conv_in" not in n],
                "lr": learning_rate_vae / vae_ch,
            },
            # 对于VAE中包含"conv_in"的参数，学习率为1e-4
            {
                "params": [p for n, p in vae.named_parameters() if "conv_in" in n],
                "lr": 1e-4,
            },
        ],
        # 设置权重衰减为1e-3
        weight_decay=1e-3,
        # 设置AdamW的beta参数为(0.9, 0.95)
        betas=(0.9, 0.95),
    )

    # 定义判别器的优化器，使用AdamW算法
    optimizer_D = optim.AdamW(
        discriminator.parameters(),  # 优化判别器的所有参数
        lr=learning_rate_disc,       # 学习率为learning_rate_disc
        weight_decay=1e-3,           # 权重衰减为1e-3
        betas=(0.9, 0.95),           # AdamW的beta参数为(0.9, 0.95)
    )

    # 初始化LPIPS（感知损失）模型，并将其移动到CUDA设备
    lpips = LPIPS().cuda()

    # 创建训练数据加载器
    dataloader = create_dataloader(
        dataset_url, batch_size, num_workers=4, do_shuffle=True
    )
    # 创建测试数据加载器
    test_dataloader = create_dataloader(
        test_dataset_url, batch_size, num_workers=4, do_shuffle=False, just_resize=True
    )

    # 计算总的训练步数
    num_training_steps = max_steps
    # 设置预热步数为200
    num_warmup_steps = 200
    # 使用余弦学习率调度器，并包含预热阶段
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer_G, num_warmup_steps, num_training_steps
    )

    # 设置日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if master_process:
        # 创建控制台处理器
        handler = logging.StreamHandler()
        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        # 将处理器添加到日志记录器
        logger.addHandler(handler)

    # 初始化全局步数为0
    global_step = 0

    # 如果提供了加载路径，则加载模型状态
    if load_path is not None:
        state_dict = torch.load(load_path, map_location="cpu")
        try:
            # 尝试严格加载VAE模型的状态
            status = vae.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(e)
            # 如果严格加载失败，则替换键名中的"_orig_mod."为""，然后再次尝试加载
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            status = vae.load_state_dict(state_dict, strict=True)
            print(status)

    # 记录开始时间
    t0 = time.time()

    # 初始化LeCam损失相关的变量
    lecam_loss_weight = 0.1
    lecam_anchor_real_logits = 0.0
    lecam_anchor_fake_logits = 0.0
    lecam_beta = 0.9

    # 开始训练循环
    for epoch in range(num_epochs):
        for i, real_images_hr in enumerate(dataloader):
            # 计算数据加载所花费的时间
            time_taken_till_load = time.time() - t0

            # 重置开始时间
            t0 = time.time()
            
            # 将高分辨率真实图像移动到指定设备（GPU）
            real_images_hr = real_images_hr[0].to(device)
            # 将高分辨率图像调整为256x256，使用面积插值方法
            real_images_for_enc = F.interpolate(
                real_images_hr, size=(256, 256), mode="area"
            )
            # 以50%的概率随机翻转图像
            if random.random() < 0.5:
                real_images_for_enc = torch.flip(real_images_for_enc, [-1])
                real_images_hr = torch.flip(real_images_hr, [-1])
            
             # 将编码前的图像输入到VAE的编码器中，获取潜在向量z
            z = vae.module.encoder(real_images_for_enc)

            # 将z从计算图中分离，并转换为CPU上的张量，然后重塑为1D
            with ctx:
                z_dist_value: torch.Tensor = z.detach().cpu().reshape(-1)

            # 定义计算峰度（Kurtosis）的函数
            def kurtosis(x):
                return ((x - x.mean()) ** 4).mean() / (x.std() ** 4)

            # 定义计算偏度（Skewness）的函数
            def skew(x):
                return ((x - x.mean()) ** 3).mean() / (x.std() ** 3)

            # 计算z的各个分位点、峰度和偏度
            z_quantiles = {
                "0.0": z_dist_value.quantile(0.0),
                "0.2": z_dist_value.quantile(0.2),
                "0.4": z_dist_value.quantile(0.4),
                "0.6": z_dist_value.quantile(0.6),
                "0.8": z_dist_value.quantile(0.8),
                "1.0": z_dist_value.quantile(1.0),
                "kurtosis": kurtosis(z_dist_value),
                "skewness": skew(z_dist_value),
            }

            # 如果需要进行钳制操作，则对z进行钳制
            if do_clamp:
                z = z.clamp(-clamp_th, clamp_th)
            
            # 对z进行正则化处理
            z_s = vae.module.reg(z)

            # 进行数据增强操作
            # 以50%的概率随机翻转z_s的最后一个维度
            if random.random() < 0.5 and flip_invariance:
                z_s = torch.flip(z_s, [-1])
                # 对特定范围进行取反
                z_s[:, -4:-2] = -z_s[:, -4:-2]
                # 同时翻转高分辨率图像的最后一个维度
                real_images_hr = torch.flip(real_images_hr, [-1])

            # 以50%的概率随机翻转z_s的倒数第二个维度
            if random.random() < 0.5 and flip_invariance:
                z_s = torch.flip(z_s, [-2])
                # 对最后两个通道进行取反
                z_s[:, -2:] = -z_s[:, -2:]
                # 同时翻转高分辨率图像的倒数第二个维度
                real_images_hr = torch.flip(real_images_hr, [-2])

            # 检查是否启用裁剪不变性
            if random.random() < 0.5 and crop_invariance:
                # 对图像和潜在向量进行裁剪操作

                # 获取z_s的高度(z_h)和宽度(z_w)
                z_h, z_w = z.shape[-2:]
                # 随机生成新的高度，范围从12到z_h - 1
                new_z_h = random.randint(12, z_h - 1)
                # 随机生成新的宽度，范围从12到z_w - 1
                new_z_w = random.randint(12, z_w - 1)
                # 随机生成高度方向的偏移量，范围从0到z_h - new_z_h - 1
                offset_z_h = random.randint(0, z_h - new_z_h - 1)
                # 随机生成宽度方向的偏移量，范围从0到z_w - new_z_w - 1
                offset_z_w = random.randint(0, z_w - new_z_w - 1)

                # 根据解码器是否也执行高分辨率操作，计算新的高度和宽度
                new_h = (
                    new_z_h * downscale_factor * 2
                    if decoder_also_perform_hr
                    else new_z_h * downscale_factor
                )
                new_w = (
                    new_z_w * downscale_factor * 2
                    if decoder_also_perform_hr
                    else new_z_w * downscale_factor
                )

                # 根据解码器是否也执行高分辨率操作，计算高度方向的偏移量
                offset_h = (
                    offset_z_h * downscale_factor * 2
                    if decoder_also_perform_hr
                    else offset_z_h * downscale_factor
                )
                # 根据解码器是否也执行高分辨率操作，计算宽度方向的偏移量
                offset_w = (
                    offset_z_w * downscale_factor * 2
                    if decoder_also_perform_hr
                    else offset_z_w * downscale_factor
                )

                # 对高分辨率真实图像进行裁剪
                real_images_hr = real_images_hr[
                    :, :, offset_h : offset_h + new_h, offset_w : offset_w + new_w
                ]
                # 对潜在向量z_s进行裁剪
                z_s = z_s[
                    :,
                    :,
                    offset_z_h : offset_z_h + new_z_h,
                    offset_z_w : offset_z_w + new_z_w,
                ]

                # 确保裁剪后的高分辨率图像和潜在向量的尺寸是否正确
                assert real_images_hr.shape[-2] == new_h
                assert real_images_hr.shape[-1] == new_w
                assert z_s.shape[-2] == new_z_h
                assert z_s.shape[-1] == new_z_w

            # 在混合精度上下文中，对潜在向量z_s进行解码，重建图像
            with ctx:
                reconstructed = vae.module.decoder(z_s)

            # 如果当前全局步数超过最大步数，则跳出循环，结束训练
            if global_step >= max_steps:
                break
            
            # 如果启用了GAN损失
            if do_ganloss:
                # 使用判别器对高分辨率真实图像进行判别
                real_preds = discriminator(real_images_hr)
                # 使用判别器对重建图像进行判别（梯度不反向传播）
                fake_preds = discriminator(reconstructed.detach())
                # 计算判别器的损失，以及平均真实和虚假逻辑值和判别准确率
                d_loss, avg_real_logits, avg_fake_logits, disc_acc = gan_disc_loss(
                    real_preds, fake_preds, disc_type
                )

                # 对节点上的平均逻辑值进行平均化处理
                avg_real_logits = avg_scalar_over_nodes(avg_real_logits, device)
                avg_fake_logits = avg_scalar_over_nodes(avg_fake_logits, device)

                # 使用LeCam方法更新真实和虚假锚点逻辑值
                lecam_anchor_real_logits = (
                    lecam_beta * lecam_anchor_real_logits
                    + (1 - lecam_beta) * avg_real_logits
                )
                lecam_anchor_fake_logits = (
                    lecam_beta * lecam_anchor_fake_logits
                    + (1 - lecam_beta) * avg_fake_logits
                )

                # 计算判别器的总损失
                total_d_loss = d_loss.mean()
                # 获取判别器损失的数值
                d_loss_item = total_d_loss.item()
                # 如果使用了LeCam方法，则添加LeCam损失
                if use_lecam:
                    # 对真实预测值与虚假锚点逻辑值之差进行平方惩罚
                    # 对虚假预测值与真实锚点逻辑值之差进行平方惩罚
                    lecam_loss = (real_preds - lecam_anchor_fake_logits).pow(
                        2
                    ).mean() + (fake_preds - lecam_anchor_real_logits).pow(2).mean()

                    # 获取LeCam损失的数值
                    lecam_loss_item = lecam_loss.item()
                    # 将LeCam损失乘以权重后加到总判别器损失中
                    total_d_loss = total_d_loss + lecam_loss * lecam_loss_weight

                # 判别器优化器的梯度归零
                optimizer_D.zero_grad()
                # 反向传播计算判别器的梯度
                total_d_loss.backward(retain_graph=True)
                # 更新判别器的参数
                optimizer_D.step()

            # 对重建图像进行反归一化处理，并计算感知损失
            _recon_for_perceptual = gradnorm(reconstructed)

            # 如果在计算感知损失之前进行数据增强
            if augment_before_perceptual_loss:
                # 克隆高分辨率真实图像
                real_images_hr_aug = real_images_hr.clone()
                # 以50%的概率随机翻转重建图像和真实图像的最后一个维度
                if random.random() < 0.5:
                    _recon_for_perceptual = torch.flip(_recon_for_perceptual, [-1])
                    real_images_hr_aug = torch.flip(real_images_hr_aug, [-1])
                # 以50%的概率随机翻转重建图像和真实图像的倒数第二个维度
                if random.random() < 0.5:
                    _recon_for_perceptual = torch.flip(_recon_for_perceptual, [-2])
                    real_images_hr_aug = torch.flip(real_images_hr_aug, [-2])

            else:
                # 如果不进行数据增强，则直接使用高分辨率真实图像
                real_images_hr_aug = real_images_hr

            # 计算感知重建损失，使用LPIPS模型计算感知相似度
            percep_rec_loss = lpips(_recon_for_perceptual, real_images_hr_aug).mean()

            # mse, vae loss.
            # 计算均方误差（MSE）损失和变分自编码器（VAE）损失。
            # 对重建图像进行梯度归一化处理，权重设为0.001
            recon_for_mse = gradnorm(reconstructed, weight=0.001)
            # 计算VAE的总损失，包括重建损失和KL散度损失
            vae_loss, loss_data = vae_loss_function(real_images_hr, recon_for_mse, z)
            
            # gan loss
            # 如果启用了GAN损失并且当前全局步数大于等于0
            if do_ganloss and global_step >= 0:
                # 对重建图像进行梯度归一化处理，权重设为1.0
                recon_for_gan = gradnorm(reconstructed, weight=1.0)
                # 使用判别器对归一化后的重建图像进行判别
                fake_preds = discriminator(recon_for_gan)
                # 克隆真实预测值并从计算图中分离，以避免梯度反向传播
                real_preds_const = real_preds.clone().detach()

                # loss where (real > fake + 0.01)
                # g_gan_loss = (real_preds_const - fake_preds - 0.1).relu().mean()
                # 计算生成器的GAN损失：
                # 如果判别器类型为二元交叉熵（BCE），则使用BCE损失函数
                if disc_type == "bce":
                    g_gan_loss = nn.functional.binary_cross_entropy_with_logits(
                        fake_preds, torch.ones_like(fake_preds)
                    )
                # 如果判别器类型为Hinge，则使用Hinge损失函数
                elif disc_type == "hinge":
                    g_gan_loss = -fake_preds.mean()

                # 计算VAE的总损失，包括感知重建损失、GAN损失和VAE损失
                overall_vae_loss = percep_rec_loss + g_gan_loss + vae_loss
                # 将GAN损失转换为数值类型，以便记录或打印
                g_gan_loss = g_gan_loss.item()
            else:
                # 如果未启用GAN损失，则VAE的总损失仅包括感知重建损失和VAE损失
                overall_vae_loss = percep_rec_loss + vae_loss
                # GAN损失设为0
                g_gan_loss = 0.0

            # 反向传播计算VAE的总损失
            overall_vae_loss.backward()
            # 更新生成器（VAE）的参数
            optimizer_G.step()
            # 清零生成器优化器的梯度
            optimizer_G.zero_grad()
            # 更新学习率调度器
            lr_scheduler.step()

            # 如果启用了GAN损失
            if do_ganloss:
                # 清零判别器优化器的梯度
                optimizer_D.zero_grad()

            # 计算从开始到当前步骤所花费的时间
            time_taken_till_step = time.time() - t0

            # 如果当前进程是主进程
            if master_process:
                # 每5个全局步记录一次日志
                if global_step % 5 == 0:
                    wandb.log(
                        {
                            "epoch": epoch,   # 当前训练周期
                            "batch": i,   # 当前批次
                            "overall_vae_loss": overall_vae_loss.item(),  # VAE总损失
                            "mse_loss": loss_data["recon_loss"],  # 均方误差损失
                            "kl_loss": loss_data["kl_loss"],  # KL散度损失
                            "perceptual_loss": percep_rec_loss.item(),  # 感知损失
                            "gan/generator_gan_loss": (  
                                g_gan_loss if do_ganloss else None  # 生成器的GAN损失
                            ),
                            "z_quantiles/abs_z": loss_data["average_of_abs_z"],  # z的绝对值的平均值
                            "z_quantiles/std_z": loss_data["std_of_abs_z"],  # z的绝对值的标准差
                            "z_quantiles/logvar": loss_data["average_of_logvar"],  # logvar的平均值
                            "gan/avg_real_logits": (
                                avg_real_logits if do_ganloss else None  # 平均真实逻辑值
                            ),
                            "gan/avg_fake_logits": (
                                avg_fake_logits if do_ganloss else None  # 平均虚假逻辑值
                            ),
                            "gan/discriminator_loss": (
                                d_loss_item if do_ganloss else None  # 判别器的损失
                            ),
                            "gan/discriminator_accuracy": (
                                disc_acc if do_ganloss else None  # 判别器的准确率
                            ),
                            "gan/lecam_loss": lecam_loss_item if do_ganloss else None,  # LeCam损失
                            "gan/lecam_anchor_real_logits": (
                                lecam_anchor_real_logits if do_ganloss else None  # LeCam真实锚点逻辑值
                            ),
                            "gan/lecam_anchor_fake_logits": (
                                lecam_anchor_fake_logits if do_ganloss else None  # LeCam虚假锚点逻辑值
                            ),
                            "z_quantiles/qs": z_quantiles,  # z的分位点
                            "time_taken_till_step": time_taken_till_step,  # 当前步骤所花费的时间
                            "time_taken_till_load": time_taken_till_load,  # 数据加载所花费的时间
                        }
                    )

                # 每200个全局步记录一次详细的损失日志
                if global_step % 200 == 0:

                    wandb.log(
                        {
                            f"loss_stepwise/mse_loss_{global_step}": loss_data[
                                "recon_loss"
                            ],  # 均方误差损失
                            f"loss_stepwise/kl_loss_{global_step}": loss_data[
                                "kl_loss"
                            ],  # KL散度损失
                            f"loss_stepwise/overall_vae_loss_{global_step}": overall_vae_loss.item(),  # VAE总损失
                        }  
                    )

                # 构建日志消息
                log_message = f"Epoch [{epoch}/{num_epochs}] - "
                # 收集日志项
                log_items = [
                    ("perceptual_loss", percep_rec_loss.item()),  # 感知损失
                    ("mse_loss", loss_data["recon_loss"]),  # 均方误差损失
                    ("kl_loss", loss_data["kl_loss"]),  # KL散度损失
                    ("overall_vae_loss", overall_vae_loss.item()),  # VAE总损失
                    ("ABS mu (0.0): average_of_abs_z", loss_data["average_of_abs_z"]),  # z绝对值的平均值
                    ("STD mu : std_of_abs_z", loss_data["std_of_abs_z"]),  # z绝对值的标准差
                    (
                        "ABS logvar (0.0) : average_of_logvar",
                        loss_data["average_of_logvar"],  # logvar的平均值
                    ),
                    ("STD logvar : std_of_logvar", loss_data["std_of_logvar"]),  # logvar的标准差
                    *[(f"z_quantiles/{q}", v) for q, v in z_quantiles.items()],  # z的分位点
                    ("time_taken_till_step", time_taken_till_step),  # 当前步骤所花费的时间
                    ("time_taken_till_load", time_taken_till_load),  # 数据加载所花费的时间
                ]

                # 如果启用了GAN损失，则添加GAN相关的日志项
                if do_ganloss:
                    log_items = [
                        ("d_loss", d_loss_item),  # 判别器损失
                        ("gan_loss", g_gan_loss),  # 生成器GAN损失
                        ("avg_real_logits", avg_real_logits),  # 平均真实逻辑值
                        ("avg_fake_logits", avg_fake_logits),  # 平均虚假逻辑值
                        ("discriminator_accuracy", disc_acc),  # 判别器准确率
                        ("lecam_loss", lecam_loss_item),  # LeCam损失
                        ("lecam_anchor_real_logits", lecam_anchor_real_logits),  # LeCam真实锚点逻辑值
                        ("lecam_anchor_fake_logits", lecam_anchor_fake_logits),  # LeCam虚假锚点逻辑值
                    ] + log_items

                # 将日志项格式化为字符串
                log_message += "\n\t".join(
                    [f"{key}: {value:.4f}" for key, value in log_items]
                )
                # 记录日志
                logger.info(log_message)

            # 增加全局步数
            global_step += 1
            # 重置开始时间
            t0 = time.time()

            # 如果满足以下条件：
            # 1. evaluate_every_n_steps 大于0
            # 2. 当前全局步数是 evaluate_every_n_steps 的倍数（从1开始计数）
            # 3. 当前进程是主进程
            if (
                evaluate_every_n_steps > 0
                and global_step % evaluate_every_n_steps == 1
                and master_process
            ):

                # 禁用梯度计算，以节省显存和加速计算
                with torch.no_grad():
                    # 初始化存储测试图像和重建图像的列表
                    all_test_images = []
                    all_reconstructed_test = []

                    # 遍历测试数据加载器中的每个批次
                    for test_images in test_dataloader:
                        # 将测试图像移动到指定设备（GPU）
                        test_images_ori = test_images[0].to(device)
                        # 将测试图像调整为256x256，使用面积插值方法
                        test_images = F.interpolate(
                            test_images_ori, size=(256, 256), mode="area"
                        )
                        # 在混合精度上下文中，使用VAE的编码器对调整后的测试图像进行编码，得到潜在向量z
                        with ctx:
                            z = vae.module.encoder(test_images)

                        # 如果需要进行钳制操作，则对z进行钳制
                        if do_clamp:
                            z = z.clamp(-clamp_th, clamp_th)

                        # 对潜在向量z进行正则化处理，得到z_s
                        z_s = vae.module.reg(z)

                        # 如果启用了翻转不变性，则对z_s进行翻转操作：
                        # 1. 对最后两个维度进行翻转
                        # 2. 对最后四个通道进行取反

                        # [1, 2]
                        # [3, 4]
                        # ->
                        # [3, 4]
                        # [1, 2]
                        # ->
                        # [4, 3]
                        # [2, 1]
                        if flip_invariance:
                            z_s = torch.flip(z_s, [-1, -2])
                            z_s[:, -4:] = -z_s[:, -4:]

                        # 在混合精度上下文中，使用VAE的解码器对z_s进行解码，重建测试图像
                        with ctx:
                            reconstructed_test = vae.module.decoder(z_s)

                        # 对原始测试图像和重建图像进行反归一化处理（假设之前进行了0.5的归一化）
                        test_images_ori = test_images_ori * 0.5 + 0.5
                        reconstructed_test = reconstructed_test * 0.5 + 0.5

                        # clamp
                        # 对图像进行钳制，确保像素值在0到1之间
                        test_images_ori = test_images_ori.clamp(0, 1)
                        reconstructed_test = reconstructed_test.clamp(0, 1)

                        # flip twice
                        # 如果启用了翻转不变性，则对重建图像进行两次翻转操作，以恢复原始方向
                        if flip_invariance:
                            reconstructed_test = torch.flip(
                                reconstructed_test, [-1, -2]
                            )

                        # 将处理后的测试图像和重建图像添加到对应的列表中
                        all_test_images.append(test_images_ori)
                        all_reconstructed_test.append(reconstructed_test)

                        # 如果收集的测试图像数量达到2张，则跳出循环（假设只处理前两批数据）
                        if len(all_test_images) >= 2:
                            break
                    
                    # 将所有测试图像和重建图像在批次维度上进行拼接
                    test_images = torch.cat(all_test_images, dim=0)
                    reconstructed_test = torch.cat(all_reconstructed_test, dim=0)

                    # 在控制台记录当前周期和测试图像的日志
                    logger.info(f"Epoch [{epoch}/{num_epochs}] - Logging test images")

                    # crop test and recon to 64 x 64
                    # 如果解码器也执行高分辨率操作，则D设为512，否则设为256
                    D = 512 if decoder_also_perform_hr else 256
                    # 设置裁剪偏移量为0
                    offset = 0
                    # 对测试图像和重建图像进行裁剪，裁剪大小为D x D
                    test_images = test_images[
                        :, :, offset : offset + D, offset : offset + D
                    ].cpu()
                    reconstructed_test = reconstructed_test[
                        :, :, offset : offset + D, offset : offset + D
                    ].cpu()

                    # concat the images into one large image.
                    # make size of (D * 4) x (D * 4)
                    # 初始化用于拼接图像的张量
                    recon_all_image = torch.zeros((3, D * 4, D * 4))
                    test_all_image = torch.zeros((3, D * 4, D * 4))

                    # 将重建图像和测试图像按2行4列的格式拼接成一个大图像
                    for i in range(2):
                        for j in range(4):
                            recon_all_image[
                                :, i * D : (i + 1) * D, j * D : (j + 1) * D
                            ] = reconstructed_test[i * 4 + j]
                            test_all_image[
                                :, i * D : (i + 1) * D, j * D : (j + 1) * D
                            ] = test_images[i * 4 + j]

                    # 使用W&B记录重建图像和测试图像
                    wandb.log(
                        {
                            "reconstructed_test_images": [
                                wandb.Image(recon_all_image),
                            ],
                            "test_images": [
                                wandb.Image(test_all_image),
                            ],
                        }
                    )

                    # 创建保存模型状态的目录
                    os.makedirs(f"./ckpt/{run_name}", exist_ok=True)
                    # 保存VAE的模型状态到指定路径
                    torch.save(
                        vae.state_dict(),
                        f"./ckpt/{run_name}/vae_epoch_{epoch}_step_{global_step}.pt",
                    )
                    # 在控制台打印保存路径
                    print(
                        f"Saved checkpoint to ./ckpt/{run_name}/vae_epoch_{epoch}_step_{global_step}.pt"
                    )

    # 调用清理函数（假设用于释放资源）
    cleanup()


if __name__ == "__main__":

    # 示例：使用torchrun启动8个进程来运行训练脚本
    # torchrun --nproc_per_node=8 vae_trainer.py
    train_ddp()
