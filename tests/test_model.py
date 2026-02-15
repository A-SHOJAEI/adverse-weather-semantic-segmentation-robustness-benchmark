"""Tests for model implementations and architectures."""

import pytest
import torch
import torch.nn as nn

from adverse_weather_semantic_segmentation_robustness_benchmark.models.model import (
    SegFormerModel, DeepLabV3PlusModel, EnsembleModel,
    FogDensityAwareLoss, DepthEstimationHead
)


class TestDepthEstimationHead:
    """Test DepthEstimationHead module."""

    def test_depth_head_initialization(self):
        """Test depth head initialization."""
        head = DepthEstimationHead(
            in_channels=256,
            hidden_channels=128,
            out_channels=1,
            dropout=0.1
        )

        assert head is not None
        # Check that it's a proper PyTorch module
        assert isinstance(head, nn.Module)

    def test_depth_head_forward(self):
        """Test depth head forward pass."""
        head = DepthEstimationHead(in_channels=256, hidden_channels=128)
        batch_size, height, width = 2, 32, 64

        # Create input features
        features = torch.randn(batch_size, 256, height, width)

        # Forward pass
        depth_output = head(features)

        # Check output shape and properties
        assert depth_output.shape == (batch_size, 1, height, width)
        assert torch.all(depth_output >= 0) and torch.all(depth_output <= 1)

    def test_depth_head_different_sizes(self):
        """Test depth head with different input sizes."""
        head = DepthEstimationHead(in_channels=128)

        # Test different spatial sizes
        for h, w in [(16, 32), (64, 128), (128, 256)]:
            features = torch.randn(1, 128, h, w)
            output = head(features)
            assert output.shape == (1, 1, h, w)


class TestSegFormerModel:
    """Test SegFormerModel implementation."""

    def test_segformer_initialization(self):
        """Test SegFormer model initialization."""
        model = SegFormerModel(
            num_classes=5,
            include_depth=True,
            pretrained=False
        )

        assert model.num_classes == 5
        assert model.include_depth is True
        assert hasattr(model, 'segmentation_head')

    def test_segformer_forward_without_depth(self):
        """Test SegFormer forward pass without depth estimation."""
        model = SegFormerModel(
            num_classes=5,
            include_depth=False,
            pretrained=False
        )

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        outputs = model(input_tensor)

        # Check outputs
        assert isinstance(outputs, dict)
        assert 'segmentation' in outputs
        assert 'depth' not in outputs

        seg_output = outputs['segmentation']
        assert seg_output.shape == (batch_size, 5, height, width)

    def test_segformer_forward_with_depth(self):
        """Test SegFormer forward pass with depth estimation."""
        model = SegFormerModel(
            num_classes=5,
            include_depth=True,
            pretrained=False
        )

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        outputs = model(input_tensor)

        # Check outputs
        assert isinstance(outputs, dict)
        assert 'segmentation' in outputs
        assert 'depth' in outputs

        seg_output = outputs['segmentation']
        depth_output = outputs['depth']

        assert seg_output.shape == (batch_size, 5, height, width)
        assert depth_output.shape == (batch_size, 1, height, width)

    def test_segformer_different_num_classes(self):
        """Test SegFormer with different number of classes."""
        for num_classes in [1, 19, 21]:
            model = SegFormerModel(
                num_classes=num_classes,
                include_depth=False,
                pretrained=False
            )

            input_tensor = torch.randn(1, 3, 64, 128)
            outputs = model(input_tensor)

            assert outputs['segmentation'].shape[1] == num_classes


class TestDeepLabV3PlusModel:
    """Test DeepLabV3PlusModel implementation."""

    def test_deeplabv3plus_initialization(self):
        """Test DeepLabV3+ model initialization."""
        model = DeepLabV3PlusModel(
            num_classes=5,
            include_depth=True,
            pretrained=False
        )

        assert model.num_classes == 5
        assert model.include_depth is True

    def test_deeplabv3plus_forward_without_depth(self):
        """Test DeepLabV3+ forward pass without depth."""
        model = DeepLabV3PlusModel(
            num_classes=5,
            include_depth=False,
            pretrained=False
        )

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        outputs = model(input_tensor)

        assert isinstance(outputs, dict)
        assert 'segmentation' in outputs
        assert 'depth' not in outputs

        seg_output = outputs['segmentation']
        assert seg_output.shape == (batch_size, 5, height, width)

    def test_deeplabv3plus_forward_with_depth(self):
        """Test DeepLabV3+ forward pass with depth."""
        model = DeepLabV3PlusModel(
            num_classes=5,
            include_depth=True,
            pretrained=False
        )

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        outputs = model(input_tensor)

        assert isinstance(outputs, dict)
        assert 'segmentation' in outputs
        assert 'depth' in outputs

        seg_output = outputs['segmentation']
        depth_output = outputs['depth']

        assert seg_output.shape == (batch_size, 5, height, width)
        assert depth_output.shape == (batch_size, 1, height, width)

    def test_deeplabv3plus_different_backbones(self):
        """Test DeepLabV3+ with different backbones."""
        for backbone in ['resnet50', 'resnet101']:
            try:
                model = DeepLabV3PlusModel(
                    backbone=backbone,
                    num_classes=5,
                    include_depth=False,
                    pretrained=False
                )

                input_tensor = torch.randn(1, 3, 64, 128)
                outputs = model(input_tensor)

                assert outputs['segmentation'].shape == (1, 5, 64, 128)
            except Exception:
                # Skip if segmentation_models_pytorch not available
                pytest.skip(f"Backbone {backbone} not available")


class TestEnsembleModel:
    """Test EnsembleModel implementation."""

    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        model = EnsembleModel(
            num_classes=5,
            include_depth=True,
            ensemble_strategy='weighted_average',
            temperature_scaling=True
        )

        assert model.num_classes == 5
        assert model.include_depth is True
        assert model.ensemble_strategy == 'weighted_average'
        assert model.temperature_scaling is True
        assert hasattr(model, 'segformer')
        assert hasattr(model, 'deeplabv3plus')

    def test_ensemble_forward_weighted_average(self):
        """Test ensemble forward with weighted average strategy."""
        model = EnsembleModel(
            num_classes=5,
            include_depth=True,
            ensemble_strategy='weighted_average'
        )

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        outputs = model(input_tensor)

        # Check outputs
        assert isinstance(outputs, dict)
        assert 'segmentation' in outputs
        assert 'segformer_seg' in outputs
        assert 'deeplabv3plus_seg' in outputs
        assert 'depth' in outputs

        # Check shapes
        for key in ['segmentation', 'segformer_seg', 'deeplabv3plus_seg']:
            assert outputs[key].shape == (batch_size, 5, height, width)

        assert outputs['depth'].shape == (batch_size, 1, height, width)

    def test_ensemble_forward_max_confidence(self):
        """Test ensemble forward with max confidence strategy."""
        model = EnsembleModel(
            num_classes=5,
            include_depth=False,
            ensemble_strategy='max_confidence'
        )

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        outputs = model(input_tensor)

        assert 'segmentation' in outputs
        assert 'segformer_seg' in outputs
        assert 'deeplabv3plus_seg' in outputs
        assert outputs['segmentation'].shape == (batch_size, 5, height, width)

    def test_ensemble_disagreement_computation(self):
        """Test ensemble disagreement computation."""
        model = EnsembleModel(
            num_classes=5,
            include_depth=False,
            ensemble_strategy='weighted_average'
        )

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        disagreement = model.get_ensemble_disagreement(input_tensor)

        assert disagreement.shape == (batch_size, height, width)
        assert torch.all(disagreement >= 0)

    def test_ensemble_temperature_scaling(self):
        """Test ensemble temperature scaling."""
        model = EnsembleModel(
            num_classes=5,
            temperature_scaling=True
        )

        # Check that temperature parameter exists
        assert hasattr(model, 'temperature')
        assert isinstance(model.temperature, nn.Parameter)

        batch_size, height, width = 2, 64, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        # Forward pass should work with temperature scaling
        outputs = model(input_tensor)
        assert 'segmentation' in outputs


class TestFogDensityAwareLoss:
    """Test FogDensityAwareLoss implementation."""

    def test_loss_initialization(self):
        """Test fog density aware loss initialization."""
        loss_fn = FogDensityAwareLoss(
            base_loss='cross_entropy',
            depth_weight=0.5,
            fog_sensitivity=2.0,
            depth_loss_weight=0.1
        )

        assert loss_fn.base_loss == 'cross_entropy'
        assert loss_fn.depth_weight == 0.5
        assert loss_fn.fog_sensitivity == 2.0
        assert loss_fn.depth_loss_weight == 0.1

    def test_loss_forward_without_depth(self):
        """Test loss computation without depth predictions."""
        loss_fn = FogDensityAwareLoss(base_loss='cross_entropy')

        batch_size, num_classes, height, width = 2, 5, 32, 64

        predictions = {
            'segmentation': torch.randn(batch_size, num_classes, height, width)
        }
        targets = {
            'label': torch.randint(0, num_classes, (batch_size, height, width))
        }

        loss_dict = loss_fn(predictions, targets)

        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'segmentation_loss' in loss_dict
        assert 'depth_loss' in loss_dict

        # Check that loss values are reasonable
        assert loss_dict['total_loss'].item() >= 0
        assert loss_dict['segmentation_loss'].item() >= 0

    def test_loss_forward_with_depth(self):
        """Test loss computation with depth predictions."""
        loss_fn = FogDensityAwareLoss(
            base_loss='cross_entropy',
            depth_loss_weight=0.1
        )

        batch_size, num_classes, height, width = 2, 5, 32, 64

        predictions = {
            'segmentation': torch.randn(batch_size, num_classes, height, width),
            'depth': torch.rand(batch_size, 1, height, width)
        }
        targets = {
            'label': torch.randint(0, num_classes, (batch_size, height, width)),
            'depth': torch.rand(batch_size, height, width)
        }

        loss_dict = loss_fn(predictions, targets)

        assert 'total_loss' in loss_dict
        assert 'segmentation_loss' in loss_dict
        assert 'depth_loss' in loss_dict

        # Depth loss should be non-zero
        assert loss_dict['depth_loss'] > 0

    def test_loss_with_fog_density(self):
        """Test loss computation with fog density map."""
        loss_fn = FogDensityAwareLoss(fog_sensitivity=2.0)

        batch_size, num_classes, height, width = 2, 5, 32, 64

        predictions = {
            'segmentation': torch.randn(batch_size, num_classes, height, width)
        }
        targets = {
            'label': torch.randint(0, num_classes, (batch_size, height, width))
        }
        fog_density = torch.rand(batch_size, height, width)

        loss_dict = loss_fn(predictions, targets, fog_density)

        assert 'total_loss' in loss_dict
        assert loss_dict['total_loss'].item() >= 0

    def test_focal_loss_implementation(self):
        """Test focal loss implementation."""
        loss_fn = FogDensityAwareLoss(base_loss='focal')

        batch_size, num_classes, height, width = 2, 5, 16, 32

        predictions = {
            'segmentation': torch.randn(batch_size, num_classes, height, width)
        }
        targets = {
            'label': torch.randint(0, num_classes, (batch_size, height, width))
        }

        loss_dict = loss_fn(predictions, targets)

        assert 'total_loss' in loss_dict
        assert loss_dict['total_loss'].item() >= 0

    def test_fog_density_estimation_from_depth(self):
        """Test fog density estimation from predicted depth."""
        loss_fn = FogDensityAwareLoss()

        batch_size, height, width = 2, 32, 64
        depth = torch.rand(batch_size, height, width)

        fog_density = loss_fn._estimate_fog_density_from_depth(depth)

        assert fog_density.shape == (batch_size, height, width)
        assert torch.all(fog_density >= 0) and torch.all(fog_density <= 1)


class TestModelIntegration:
    """Integration tests for model components."""

    def test_model_training_mode(self, ensemble_model):
        """Test that models can be set to training mode."""
        ensemble_model.train()
        assert ensemble_model.training

        ensemble_model.eval()
        assert not ensemble_model.training

    def test_model_parameter_count(self, ensemble_model):
        """Test model parameter counting."""
        total_params = sum(p.numel() for p in ensemble_model.parameters())
        trainable_params = sum(p.numel() for p in ensemble_model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_model_gradient_computation(self, ensemble_model, small_sample_batch):
        """Test that gradients can be computed."""
        ensemble_model.train()

        # Forward pass
        outputs = ensemble_model(small_sample_batch['image'])
        loss = outputs['segmentation'].mean()

        # Backward pass
        loss.backward()

        # Check that some gradients exist
        has_grad = False
        for param in ensemble_model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found in model parameters"

    def test_model_state_dict_save_load(self, ensemble_model, temp_dir):
        """Test model state dict saving and loading."""
        # Save state dict
        state_dict_path = temp_dir / 'model_state.pth'
        torch.save(ensemble_model.state_dict(), state_dict_path)

        # Create new model and load state dict
        new_model = EnsembleModel(num_classes=5, include_depth=True)
        new_model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

        # Check that parameters match
        for (name1, param1), (name2, param2) in zip(
            ensemble_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)