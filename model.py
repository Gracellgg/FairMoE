from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

#ResidualBlock
class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        # self.left = nn.Sequential(
        #     nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
        #     nn.BatchNorm2d(outchannel),
        #     nn.ReLU(),
        #     nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(outchannel)
        # )
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.downsample = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        #out = self.left(x) #torch.Size([1, 64, 128, 128])
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print('residual block 0')
        # print(out.shape)
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64  # 输入的像素为64
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 三通道，三个颜色输入
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.avgpool = nn.AvgPool2d(kernel_size=4)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1] or [2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, _):
        out = self.conv1(x) # torch.Size([1, 3, 128, 128])
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        #out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out, None

    def set_weights(self, weights):
        self.load_state_dict(weights)





class MoEConv(nn.Module):
    """Mixture of Experts Convolutional layer.
    
    This layer implements a mixture of experts approach where multiple convolutional
    layers (experts) are combined using a router network.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        stride: Stride for the convolution
        padding: Padding for the convolution
        n_experts: Number of expert networks
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, n_experts: int = 2) -> None:
        super().__init__()
        self.n_experts = n_experts
        
        # Expert networks
        self.expert_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            for _ in range(n_experts)
        ])

        # Router network for training (with sensitive attribute)
        self.router_train = nn.Sequential(
            nn.Linear(in_channels + 1, 128),  # +1 for sensitive attribute
            nn.ReLU(),
            nn.Linear(128, n_experts),
            nn.Softmax(dim=1)
        )
        
        # Router network for testing (without sensitive attribute)
        self.router_test = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, n_experts),
            nn.Softmax(dim=1)
        )
        
        # Initialize balancing parameters as tensors
        self.register_buffer('alpha_k', torch.ones(n_experts))
        self.register_buffer('group_counts', torch.zeros(n_experts))
        self.register_buffer('total_samples', torch.tensor(0, dtype=torch.long))

    def update_alpha_k(self, sensitive_attribute: torch.Tensor) -> None:
        """Update balancing parameters based on group sizes.
        
        Args:
            sensitive_attribute: Tensor containing sensitive attribute values
        """
        # Update group counts
        for k in range(self.n_experts):
            self.group_counts[k] = (sensitive_attribute == k).sum()
        
        # Update total samples
        self.total_samples.fill_(sensitive_attribute.size(0))
        
        # Update alpha_k with epsilon to avoid division by zero
        self.alpha_k = 1.0 / (self.group_counts + 1e-6)

    def calculate_mi_loss(self, scores_soft: torch.Tensor, 
                         sensitive_attribute: torch.Tensor) -> torch.Tensor:
        """Calculate mutual information loss.
        
        Args:
            scores_soft: Soft router scores
            sensitive_attribute: Tensor containing sensitive attribute values
            
        Returns:
            Mutual information loss
        """
        batch_size = scores_soft.size(0)

        # Calculate joint probability
        joint_prob = torch.zeros(self.n_experts, self.n_experts, device=scores_soft.device)
        for i in range(self.n_experts):
            for j in range(self.n_experts):
                mask = (sensitive_attribute == i)
                joint_prob[i, j] = (scores_soft[mask, j].sum() / batch_size)
        
        # Calculate marginal probabilities
        p_c = joint_prob.sum(dim=1, keepdim=True)
        p_e = joint_prob.sum(dim=0, keepdim=True)
        
        # Calculate mutual information
        eps = 1e-10
        mi = torch.sum(joint_prob * torch.log((joint_prob + eps) / (p_c * p_e + eps)))
        
        # Calculate positive correlation loss
        positive_correlation_loss = sum(
            torch.mean((sensitive_attribute == i).float() * scores_soft[:, i])
            for i in range(self.n_experts)
        )
        
        return -mi #+ positive_correlation_loss

    def forward(self, x: torch.Tensor, sensitive_attribute: Optional[torch.Tensor] = None, 
                train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
            sensitive_attribute: Tensor containing sensitive attribute values (only needed during training)
            train: Whether in training mode
            
        Returns:
            Tuple of (output tensor, mutual information loss, router scores)
        """
        batch_size = x.size(0)
        x_flat = x.mean(dim=[2, 3])
        
        if train and sensitive_attribute is not None:
            # Training mode with sensitive attribute
            router_input = torch.cat([x_flat, sensitive_attribute.view(-1, 1)], dim=1)
            router_scores = self.router_train(router_input)
            self.update_alpha_k(sensitive_attribute)
            
            # Apply balancing
            balanced_scores = router_scores * self.alpha_k.view(1, -1)
            router_weights = balanced_scores / (balanced_scores.sum(dim=1, keepdim=True) + 1e-6)
            
            # Calculate mutual information loss
            mi_loss = self.calculate_mi_loss(router_weights, sensitive_attribute)
        else:
            # Testing mode or no sensitive attribute provided
            router_scores = self.router_test(x_flat)
            router_weights = router_scores
            mi_loss = torch.tensor(0.0, device=x.device)
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.expert_convs], dim=1)
        
        # Combine expert outputs
        router_weights = router_weights.view(batch_size, self.n_experts, 1, 1, 1)
        out = torch.sum(expert_outputs * router_weights, dim=1)
        
        return out, mi_loss, router_weights.squeeze(-1).squeeze(-1).squeeze(-1)

class MoEResidualBlock(nn.Module):
    """Residual block with Mixture of Experts.
    
    Args:
        inchannel: Number of input channels
        outchannel: Number of output channels
        stride: Stride for the first convolution
    """
    def __init__(self, inchannel: int, outchannel: int, stride: int = 1) -> None:
        super().__init__()
        self.moe_conv1 = MoEConv(inchannel, outchannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        self.moe_conv2 = MoEConv(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)
        
        self.downsample = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x: torch.Tensor, sensitive_attribute: Optional[torch.Tensor] = None, 
                train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        identity = x
        
        out, mi_loss1, router_scores1 = self.moe_conv1(x, sensitive_attribute, train)
        out = self.bn1(out)
        out = self.relu(out)
        
        out, mi_loss2, router_scores2 = self.moe_conv2(out, sensitive_attribute, train)
        out = self.bn2(out)
        
        out += self.downsample(identity)
        out = self.relu(out)
        
        return out, mi_loss1 + mi_loss2, [router_scores1, router_scores2]

class MoEResNet(nn.Module):
    """ResNet architecture with Mixture of Experts.
    
    Args:
        block: Type of residual block to use
        num_classes: Number of output classes
    """
    def __init__(self, block: nn.Module, num_classes: int = 10) -> None:
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Create layers without Sequential wrapper
        self.layer1 = self.make_layer(block, 64, 2, stride=1)
        self.layer2 = self.make_layer(block, 128, 2, stride=2)
        self.layer3 = self.make_layer(block, 256, 2, stride=2)
        self.layer4 = self.make_layer(block, 512, 2, stride=2)
        
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block: nn.Module, channels: int, num_blocks: int, 
                   stride: int) -> nn.ModuleList:
        """Create a layer of residual blocks.
        
        Args:
            block: Type of residual block to use
            channels: Number of output channels
            num_blocks: Number of blocks in the layer
            stride: Stride for the first block
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, sensitive_attribute: Optional[torch.Tensor] = None, 
                train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
            sensitive_attribute: Tensor containing sensitive attribute values (only needed during training)
            train: Whether in training mode
            
        Returns:
            Tuple of (output tensor, total mutual information loss, router scores)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        router_scores = []
        mi_loss_total = 0

        # Process each layer
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                out, mi_loss, scores = block(out, sensitive_attribute, train)
                mi_loss_total += mi_loss
                router_scores.extend(scores)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        router_scores = torch.stack(router_scores, dim=1)

        return out, mi_loss_total, router_scores

def MoEResNet18(num_classes: int = 10, n_experts: int = 2) -> MoEResNet:
    """Create a MoE ResNet18 model.
    
    Args:
        num_classes: Number of output classes
        n_experts: Number of experts in the MoE model
        
    Returns:
        MoE ResNet18 model
    """
    model = MoEResNet(MoEResidualBlock, num_classes)
    # Set number of experts for all MoEConv layers
    for module in model.modules():
        if isinstance(module, MoEConv):
            module.n_experts = n_experts
            module.expert_convs = nn.ModuleList([
                nn.Conv2d(module.expert_convs[0].in_channels, 
                         module.expert_convs[0].out_channels,
                         module.expert_convs[0].kernel_size,
                         stride=module.expert_convs[0].stride,
                         padding=module.expert_convs[0].padding)
                for _ in range(n_experts)
            ])
            module.router_train = nn.Sequential(
                nn.Linear(module.router_train[0].in_features, 128),
                nn.ReLU(),
                nn.Linear(128, n_experts),
                nn.Softmax(dim=1)
            )
            module.router_test = nn.Sequential(
                nn.Linear(module.router_test[0].in_features, 128),
                nn.ReLU(),
                nn.Linear(128, n_experts),
                nn.Softmax(dim=1)
            )
            module.register_buffer('alpha_k', torch.ones(n_experts))
            module.register_buffer('group_counts', torch.zeros(n_experts))
    return model

def analyze_router_score(router_score: torch.Tensor, 
                        sensitive_attribute: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Analyze router scores for each sensitive attribute group.
    
    Args:
        router_score: Router scores tensor
        sensitive_attribute: Tensor containing sensitive attribute values
        
    Returns:
        Tuple of (score combinations, activation counts)
    """
    score_class_combination = {}
    activate_count = {}
    
    for attr in torch.unique(sensitive_attribute):
        mask = (sensitive_attribute == attr)
        if mask.sum() > 0:
            scores_attr = router_score[mask]
            score_class_combination[f'attr_{attr.item()}'] = scores_attr.mean(dim=0)
            activate_count[f'attr_{attr.item()}'] = mask.sum()
    
    return score_class_combination, activate_count
