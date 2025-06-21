import argparse, os
import shutil
# from tqdm import tqdm
# from time import sleep

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

#from model import CustomResNet18, SelectResNet18, HeadResNet18, MoEResNet18, ResNet18



def print_layer_output(name):
    def hook(module, input, output):
        print(f"Layer: {name}")
        print(f"Output: {output[0][0][0]}")
        print(f"Output Shape: {output.shape}\n")
    return hook


def copy_bn_params(src_bn: nn.BatchNorm2d, dst_bn: nn.BatchNorm2d) -> None:
    """Copy batch normalization parameters from source to destination.
    
    Args:
        src_bn: Source batch normalization layer
        dst_bn: Destination batch normalization layer
    """
    dst_bn.weight.data.copy_(src_bn.weight.data)
    dst_bn.bias.data.copy_(src_bn.bias.data)
    dst_bn.running_mean.data.copy_(src_bn.running_mean.data)
    dst_bn.running_var.data.copy_(src_bn.running_var.data)
    dst_bn.num_batches_tracked.data.copy_(src_bn.num_batches_tracked.data)


def copy_downsample_params(src_block: nn.Module, dst_block: nn.Module) -> None:
    """Copy downsample parameters from source to destination block.
    
    Args:
        src_block: Source block containing downsample layers
        dst_block: Destination block containing downsample layers
    """
    if src_block.downsample is not None:
        dst_block.downsample[0].weight.data.copy_(src_block.downsample[0].weight.data)
        copy_bn_params(src_block.downsample[1], dst_block.downsample[1])


def initialize_moe_resnet18_from_resnet18(moe_resnet: nn.Module, resnet: nn.Module) -> None:
    """Initialize MoE ResNet18 parameters from a pretrained ResNet18.
    
    This function copies all the parameters from a pretrained ResNet18 to a MoE ResNet18,
    including convolutional layers, batch normalization layers, and fully connected layers.
    
    Args:
        moe_resnet: The MoE ResNet18 model to be initialized
        resnet: The pretrained ResNet18 model
    """
    # Initialize initial convolutional layer
    moe_resnet.conv1.weight.data.copy_(resnet.conv1.weight.data)
    copy_bn_params(resnet.bn1, moe_resnet.bn1)

    # Initialize each layer
    for moe_layer, res_layer in zip(
        [moe_resnet.layer1, moe_resnet.layer2, moe_resnet.layer3, moe_resnet.layer4],
        [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    ):
        for moe_block, res_block in zip(moe_layer, res_layer):
            # Initialize expert convolutions
            if hasattr(moe_block, 'moe_conv1'):
                # Initialize all expert convolutions with the same weights
                for expert_conv in moe_block.moe_conv1.expert_convs:
                    expert_conv.weight.data.copy_(res_block.conv1.weight.data)
                for expert_conv in moe_block.moe_conv2.expert_convs:
                    expert_conv.weight.data.copy_(res_block.conv2.weight.data)
            else:
                # Fallback for non-MoE layers
                moe_block.left[0].weight.data.copy_(res_block.conv1.weight.data)
                moe_block.left[3].weight.data.copy_(res_block.conv2.weight.data)

            # Copy batch normalization parameters
            copy_bn_params(res_block.bn1, moe_block.bn1)
            copy_bn_params(res_block.bn2, moe_block.bn2)

            # Copy downsample parameters if they exist
            copy_downsample_params(res_block, moe_block)

    # Initialize final fully connected layer
    moe_resnet.fc.weight.data.copy_(resnet.fc.weight.data)
    moe_resnet.fc.bias.data.copy_(resnet.fc.bias.data)


# Set random seeds for reproducibility
def set_seed(seed: int = 0) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# Initialize with default seed
set_seed()


