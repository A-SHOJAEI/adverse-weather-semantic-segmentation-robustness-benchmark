"""Model implementations for semantic segmentation with weather robustness."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import SegformerModel, SegformerConfig
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class DepthEstimationHead(nn.Module):
    """
    Depth estimation head for multi-task learning.

    Provides depth estimates to support fog-density-aware loss computation
    and improve scene understanding.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 1,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize depth estimation head.

        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden channels
            out_channels: Number of output channels (1 for depth)
            dropout: Dropout probability
        """
        super().__init__()

        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()  # Normalize depth to [0, 1]
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for depth estimation.

        Args:
            features: Input feature tensor [B, C, H, W]

        Returns:
            Depth predictions [B, 1, H, W]
        """
        depth = self.depth_head(features)
        return depth


class SegFormerModel(nn.Module):
    """
    SegFormer model with custom depth estimation head.

    Implements SegFormer architecture with additional depth prediction
    for multi-task learning and weather robustness.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_classes: int = 19,
        include_depth: bool = True,
        pretrained: bool = True
    ) -> None:
        """
        Initialize SegFormer model.

        Args:
            model_name: Pretrained model name
            num_classes: Number of segmentation classes
            include_depth: Whether to include depth estimation
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.include_depth = include_depth

        # Load SegFormer backbone
        try:
            if pretrained:
                self.segformer = SegformerModel.from_pretrained(model_name)
            else:
                try:
                    config = SegformerConfig.from_pretrained(model_name)
                    self.segformer = SegformerModel(config)
                except Exception:
                    # If pretrained config fails, use basic config
                    config = SegformerConfig(
                        num_channels=3,
                        num_encoder_blocks=4,
                        depths=[2, 2, 2, 2],
                        sr_ratios=[8, 4, 2, 1],
                        hidden_sizes=[32, 64, 160, 256],
                        patch_sizes=[7, 3, 3, 3],
                        strides=[4, 2, 2, 2],
                        num_attention_heads=[1, 2, 5, 8],
                        mlp_ratios=[4, 4, 4, 4]
                    )
                    self.segformer = SegformerModel(config)
        except Exception as e:
            logger.warning(f"Could not load SegFormer from transformers: {e}")
            # Fallback to basic configuration
            config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[32, 64, 160, 256],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4]
            )
            self.segformer = SegformerModel(config)

        # Get feature dimensions
        self.feature_dim = getattr(self.segformer.config, 'hidden_sizes', [256])[-1]

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # Depth estimation head (optional)
        if self.include_depth:
            self.depth_head = DepthEstimationHead(
                in_channels=self.feature_dim,
                hidden_channels=128,
                out_channels=1
            )

        self._initialize_heads()
        logger.info(f"Initialized SegFormer model with {num_classes} classes")

    def _initialize_heads(self) -> None:
        """Initialize segmentation and depth heads."""
        for m in self.segmentation_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SegFormer model.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dictionary containing segmentation and depth predictions
        """
        # Extract features using SegFormer encoder
        outputs = self.segformer(x, output_hidden_states=True)

        # Use the last hidden state as features
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        else:
            # Fallback for different SegFormer versions
            features = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs[0]

        # Ensure features have the right spatial dimensions
        if len(features.shape) == 3:
            # Reshape from sequence to spatial format if needed
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).view(B, C, H, W)

        # Upsample features to match input resolution
        target_size = (x.shape[2], x.shape[3])
        features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)

        # Generate segmentation prediction
        seg_logits = self.segmentation_head(features)

        results = {'segmentation': seg_logits}

        # Generate depth prediction if enabled
        if self.include_depth:
            depth_pred = self.depth_head(features)
            results['depth'] = depth_pred

        return results


class DeepLabV3PlusModel(nn.Module):
    """
    DeepLabV3+ model with custom depth estimation.

    Implements DeepLabV3+ architecture with ASPP module and
    optional depth prediction for multi-task learning.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 19,
        include_depth: bool = True,
        pretrained: bool = True,
        output_stride: int = 16
    ) -> None:
        """
        Initialize DeepLabV3+ model.

        Args:
            backbone: Backbone architecture
            num_classes: Number of segmentation classes
            include_depth: Whether to include depth estimation
            pretrained: Whether to use pretrained backbone
            output_stride: Output stride for ASPP module
        """
        super().__init__()

        self.num_classes = num_classes
        self.include_depth = include_depth

        # Use segmentation_models_pytorch for robust implementation
        try:
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if pretrained else None,
                classes=num_classes,
                activation=None,  # Will apply softmax during training
                aux_params=None
            )

            # Get encoder feature dimension
            self.feature_dim = self.model.encoder.out_channels[-1]

        except Exception as e:
            logger.warning(f"Could not create DeepLabV3+ with smp: {e}")
            # Fallback to basic implementation
            self.model = self._create_basic_deeplabv3plus(backbone, num_classes, pretrained)
            self.feature_dim = 2048

        # Add depth estimation head if required
        if self.include_depth:
            self.depth_head = DepthEstimationHead(
                in_channels=self.feature_dim,
                hidden_channels=256,
                out_channels=1
            )

        logger.info(f"Initialized DeepLabV3+ model with {backbone} backbone")

    def _create_basic_deeplabv3plus(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool
    ) -> nn.Module:
        """Create basic DeepLabV3+ implementation as fallback."""
        # Load backbone
        if backbone == "resnet50":
            backbone_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            backbone_model = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            backbone_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048

        # Create simplified segmentation head
        classifier = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # Combine backbone and classifier
        class BasicDeepLabV3Plus(nn.Module):
            def __init__(self, backbone, classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier

            def forward(self, x):
                features = self.backbone.conv1(x)
                features = self.backbone.bn1(features)
                features = self.backbone.relu(features)
                features = self.backbone.maxpool(features)
                features = self.backbone.layer1(features)
                features = self.backbone.layer2(features)
                features = self.backbone.layer3(features)
                features = self.backbone.layer4(features)

                # Global average pooling and classification
                out = F.adaptive_avg_pool2d(features, (1, 1))
                out = self.classifier(out)
                out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
                return out

        return BasicDeepLabV3Plus(backbone_model, classifier)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DeepLabV3+ model.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dictionary containing segmentation and depth predictions
        """
        # Get segmentation prediction
        seg_logits = self.model(x)

        results = {'segmentation': seg_logits}

        # Generate depth prediction if enabled
        if self.include_depth:
            # Extract features for depth estimation
            if hasattr(self.model, 'encoder'):
                # smp model
                features = self.model.encoder(x)[-1]
            else:
                # Basic model - extract from backbone
                features = x
                for layer in [self.model.backbone.conv1, self.model.backbone.bn1,
                             self.model.backbone.relu, self.model.backbone.maxpool,
                             self.model.backbone.layer1, self.model.backbone.layer2,
                             self.model.backbone.layer3, self.model.backbone.layer4]:
                    features = layer(features)

            depth_pred = self.depth_head(features)
            # Upsample depth to match input resolution
            target_size = (x.shape[2], x.shape[3])
            depth_pred = F.interpolate(depth_pred, size=target_size, mode='bilinear', align_corners=False)
            results['depth'] = depth_pred

        return results


class EnsembleModel(nn.Module):
    """
    Ensemble model combining SegFormer and DeepLabV3+ with confidence calibration.

    Implements confidence-calibrated ensemble for improved robustness
    and uncertainty estimation in adverse weather conditions.
    """

    def __init__(
        self,
        num_classes: int = 19,
        include_depth: bool = True,
        ensemble_strategy: str = "weighted_average",
        temperature_scaling: bool = True
    ) -> None:
        """
        Initialize ensemble model.

        Args:
            num_classes: Number of segmentation classes
            include_depth: Whether to include depth estimation
            ensemble_strategy: Strategy for combining predictions
            temperature_scaling: Whether to apply temperature scaling
        """
        super().__init__()

        self.num_classes = num_classes
        self.include_depth = include_depth
        self.ensemble_strategy = ensemble_strategy
        self.temperature_scaling = temperature_scaling

        # Initialize component models
        self.segformer = SegFormerModel(
            num_classes=num_classes,
            include_depth=include_depth
        )

        self.deeplabv3plus = DeepLabV3PlusModel(
            num_classes=num_classes,
            include_depth=include_depth
        )

        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)

        # Temperature parameters for calibration
        if self.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))

        logger.info(f"Initialized ensemble model with {ensemble_strategy} strategy")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble model.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dictionary containing ensemble predictions and individual model outputs
        """
        # Get predictions from both models
        segformer_outputs = self.segformer(x)
        deeplabv3plus_outputs = self.deeplabv3plus(x)

        # Combine segmentation predictions
        if self.ensemble_strategy == "weighted_average":
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_seg = (weights[0] * segformer_outputs['segmentation'] +
                           weights[1] * deeplabv3plus_outputs['segmentation'])
        elif self.ensemble_strategy == "max_confidence":
            # Use prediction with higher maximum confidence
            seg1_conf = F.softmax(segformer_outputs['segmentation'], dim=1).max(dim=1)[0]
            seg2_conf = F.softmax(deeplabv3plus_outputs['segmentation'], dim=1).max(dim=1)[0]

            # Create selection mask
            use_segformer = (seg1_conf > seg2_conf).float().unsqueeze(1)
            ensemble_seg = (use_segformer * segformer_outputs['segmentation'] +
                           (1 - use_segformer) * deeplabv3plus_outputs['segmentation'])
        else:  # simple average
            ensemble_seg = (segformer_outputs['segmentation'] +
                           deeplabv3plus_outputs['segmentation']) / 2

        # Apply temperature scaling for calibration
        if self.temperature_scaling:
            ensemble_seg = ensemble_seg / self.temperature

        results = {
            'segmentation': ensemble_seg,
            'segformer_seg': segformer_outputs['segmentation'],
            'deeplabv3plus_seg': deeplabv3plus_outputs['segmentation']
        }

        # Combine depth predictions if available
        if self.include_depth:
            if self.ensemble_strategy == "weighted_average":
                weights = F.softmax(self.ensemble_weights, dim=0)
                ensemble_depth = (weights[0] * segformer_outputs['depth'] +
                                weights[1] * deeplabv3plus_outputs['depth'])
            else:
                ensemble_depth = (segformer_outputs['depth'] +
                                deeplabv3plus_outputs['depth']) / 2

            results.update({
                'depth': ensemble_depth,
                'segformer_depth': segformer_outputs['depth'],
                'deeplabv3plus_depth': deeplabv3plus_outputs['depth']
            })

        return results

    def get_ensemble_disagreement(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate ensemble disagreement for uncertainty estimation.

        Args:
            x: Input tensor

        Returns:
            Disagreement map [B, H, W]
        """
        with torch.no_grad():
            outputs = self.forward(x)

            # Get probabilities
            seg1_probs = F.softmax(outputs['segformer_seg'], dim=1)
            seg2_probs = F.softmax(outputs['deeplabv3plus_seg'], dim=1)

            # Calculate Jensen-Shannon divergence as disagreement measure
            mean_probs = (seg1_probs + seg2_probs) / 2

            kl1 = F.kl_div(seg1_probs.log(), mean_probs, reduction='none').sum(dim=1)
            kl2 = F.kl_div(seg2_probs.log(), mean_probs, reduction='none').sum(dim=1)

            disagreement = (kl1 + kl2) / 2

        return disagreement


class FogDensityAwareLoss(nn.Module):
    """
    Custom fog-density-aware loss function.

    Reweights pixel contributions based on estimated scene depth and fog density
    to improve model robustness in foggy conditions.
    """

    def __init__(
        self,
        base_loss: str = "cross_entropy",
        depth_weight: float = 0.5,
        fog_sensitivity: float = 2.0,
        depth_loss_weight: float = 0.1
    ) -> None:
        """
        Initialize fog-density-aware loss.

        Args:
            base_loss: Base loss function type
            depth_weight: Weight for depth-based reweighting
            fog_sensitivity: Sensitivity to fog density changes
            depth_loss_weight: Weight for depth estimation loss
        """
        super().__init__()

        self.base_loss = base_loss
        self.depth_weight = depth_weight
        self.fog_sensitivity = fog_sensitivity
        self.depth_loss_weight = depth_loss_weight

        # Base segmentation loss
        if base_loss == "cross_entropy":
            self.seg_loss_fn = nn.CrossEntropyLoss(reduction='none')
        elif base_loss == "focal":
            self.seg_loss_fn = self._focal_loss
        else:
            self.seg_loss_fn = nn.CrossEntropyLoss(reduction='none')

        # Depth loss
        self.depth_loss_fn = nn.MSELoss(reduction='none')

        logger.info(f"Initialized FogDensityAwareLoss with {base_loss} base loss")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        fog_density: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute fog-density-aware loss.

        Args:
            predictions: Model predictions containing 'segmentation' and optionally 'depth'
            targets: Ground truth targets containing 'label' and optionally 'depth'
            fog_density: Fog density map [B, H, W]

        Returns:
            Dictionary of loss components
        """
        seg_pred = predictions['segmentation']
        seg_target = targets['label'].long()

        # Compute base segmentation loss
        seg_loss = self.seg_loss_fn(seg_pred, seg_target)

        # Apply fog-density-aware weighting
        if fog_density is not None:
            # Increase loss weight in foggy regions
            fog_weight = 1.0 + self.fog_sensitivity * fog_density
            seg_loss = seg_loss * fog_weight

        # Apply depth-based weighting if depth predictions are available
        if 'depth' in predictions and self.depth_weight > 0:
            pred_depth = predictions['depth'].squeeze(1)  # Remove channel dimension

            # Estimate fog density from predicted depth if not provided
            if fog_density is None:
                fog_density = self._estimate_fog_density_from_depth(pred_depth)
                fog_weight = 1.0 + self.fog_sensitivity * fog_density
                seg_loss = seg_loss * fog_weight

            # Add depth loss if ground truth depth is available
            depth_loss = 0.0
            if 'depth' in targets:
                depth_target = targets['depth']
                depth_loss = self.depth_loss_fn(pred_depth, depth_target)
                depth_loss = depth_loss.mean()

        else:
            depth_loss = 0.0

        # Aggregate losses
        total_seg_loss = seg_loss.mean()
        total_loss = total_seg_loss + self.depth_loss_weight * depth_loss

        return {
            'total_loss': total_loss,
            'segmentation_loss': total_seg_loss,
            'depth_loss': depth_loss
        }

    def _focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.

        Args:
            inputs: Predicted logits [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            alpha: Weighting factor
            gamma: Focusing parameter

        Returns:
            Focal loss tensor
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

        return focal_loss

    def _estimate_fog_density_from_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Estimate fog density from predicted depth.

        Args:
            depth: Predicted depth map [B, H, W]

        Returns:
            Estimated fog density [B, H, W]
        """
        # Simple heuristic: fog density increases with depth
        # and varies based on local depth gradients

        # Normalize depth to [0, 1]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Base fog density proportional to depth
        fog_density = depth_norm * 0.7

        # Add fog density variations based on depth gradients
        depth_grad_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
        depth_grad_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])

        # Pad gradients to match original size
        depth_grad_x = F.pad(depth_grad_x, (0, 1, 0, 0), mode='replicate')
        depth_grad_y = F.pad(depth_grad_y, (0, 0, 0, 1), mode='replicate')

        depth_gradient_mag = torch.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1e-8)

        # High gradients indicate edges which are less affected by fog
        edge_mask = (depth_gradient_mag > depth_gradient_mag.mean()) * 0.3
        fog_density = fog_density - edge_mask

        return torch.clamp(fog_density, 0, 1)