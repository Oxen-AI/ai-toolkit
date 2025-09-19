import os
from typing import TYPE_CHECKING, Optional, List, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.accelerator import unwrap_model
from einops import rearrange, repeat

try:
    # DINOv3 requires transformers v4.56+
    from transformers import DINOv3ViTModel, DINOv3ViTImageProcessorFast, DINOv3ViTConfig
    DINOV3_AVAILABLE = True
    # Create aliases for consistency
    Dinov3Model = DINOv3ViTModel
    Dinov3ImageProcessor = DINOv3ViTImageProcessorFast
    Dinov3Config = DINOv3ViTConfig
except ImportError:
    DINOV3_AVAILABLE = False
    Dinov3Model = None
    Dinov3ImageProcessor = None
    Dinov3Config = None

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


class SegmentationDecoder(nn.Module):

    def __init__(
        self,
        embed_dims: List[int],
        num_classes: int,
        decoder_type: str = "linear",  # "linear", "mlp", "multiscale"
        target_size: int = 512,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.target_size = target_size

        if decoder_type == "linear":
            # Simple linear decoder - just use the last layer features
            # self.decoder = nn.Linear(embed_dims[-1], num_classes * (16**2))
            self.decoder = nn.Linear(embed_dims[-1], num_classes)
        elif decoder_type == "mlp":
            hidden_dim = embed_dims[-1] // 2
            self.decoder = nn.Sequential(
                nn.Linear(embed_dims[-1], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        # elif decoder_type == "multiscale":
        #     # Multi-scale feature fusion
        #     total_dim = sum(embed_dims)
        #     self.feature_align = nn.ModuleList([
        #         nn.Linear(dim, embed_dims[-1]) for dim in embed_dims
        #     ])
        #     hidden_dim = embed_dims[-1]
        #     self.decoder = nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim // 2),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.1),
        #         nn.Linear(hidden_dim // 2, num_classes)
        #     )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, images: List[torch.Tensor], features: List[torch.Tensor]) -> torch.Tensor:
        # print(images[-1].shape)
        w, h = images[-1].shape[-1], images[-1].shape[-2]
        if self.decoder_type in ["linear", "mlp"]:
            # Use only the last layer features
            patch_features = features[-1] # [B, N_patches, embed_dim]
            # patch_features = torch.cat(list(features[-i] for i in range(1, 24)), dim=-1)  # [B, N_patches, embed_dim]
            logits = self.decoder(patch_features)  # [B, N_patches, num_classes * 16 * 16] for linear

            B, N_patches, _ = logits.shape

            if self.decoder_type == "linear":
                # For linear decoder: reshape from flat to dense patch predictions
                # logits = logits.view(B, N_patches, self.num_classes, 16, 16)  # [B, N_patches, C, 16, 16]

                # Reshape and permute to get [B, num_classes, H, W]
                logits = logits.transpose(1, 2)  # [B, num_classes, N_patches]
                # print(f"logits.shape = {logits.shape}")
                h_patches = self.target_size // 16
                w_patches = (w * self.target_size) // (h * 16)
                logits = logits.view(B, -1, h_patches, w_patches)
                # print(f"logits.shape = {logits.shape}")

                # # Calculate patch grid dimensions
                # patch_h = self.target_size // 16 #int(N_patches ** 0.5)  # Assuming square patch grid
                # patch_w = (w * self.target_size) // (h * 16) #N_patches // patch_h

                # # Rearrange from patch-based to spatial layout
                # logits = logits.view(B, patch_h, patch_w, self.num_classes, 16, 16)  # [B, pH, pW, C, 16, 16]
                # logits = logits.permute(0, 3, 1, 4, 2, 5)  # [B, C, pH, 16, pW, 16]
                # logits = logits.contiguous().view(B, self.num_classes, patch_h * 16, patch_w * 16)  # [B, C, H, W]
            else:
                logits = logits.view(B, -1, self.num_classes)
                logits = logits.transpose(1, 2)  # [B, num_classes, N_patches]
                h_patches = self.target_size
                w_patches = (w * self.target_size) // h
                logits = logits.view(B, self.num_classes, h_patches, w_patches)
        # elif self.decoder_type == "multiscale":
        #     # Align all features to the same dimension and average
        #     aligned_features = []
        #     for i, feat in enumerate(features):
        #         aligned = self.feature_align[i](feat)
        #         aligned_features.append(aligned)

        #     # Average the aligned features
        #     patch_features = torch.stack(aligned_features, dim=0).mean(dim=0)
        #     logits = self.decoder(patch_features)
            
        #     # Use old reshaping for multiscale (single prediction per patch)
        #     B, N_patches, num_classes = logits.shape
        #     logits = logits.transpose(1, 2)  # [B, num_classes, N_patches]
        #     h_patches = self.target_size
        #     w_patches = (w * self.target_size) // h
        #     logits = logits.view(B, num_classes, h_patches, w_patches)


        # return TF.resize(TF.to_tensor(mask_image), [h_patches * patch_size, w_patches * patch_size])

        # Upsample to target size
        # if logits.shape[-1] != self.target_size // 16:  # Assuming 16x downsampling
        #     print(f"Resizing decoder because {logits.shape[-1]} != {self.target_size // 16}")
        #     logits = F.interpolate(
        #         logits, 
        #         size=(self.target_size // 16, self.target_size // 16), 
        #         mode='bilinear', 
        #         align_corners=False
        #     )

        # # Final upsampling to target resolution
        # logits = F.interpolate(
        #     logits,
        #     size=(self.target_size, self.target_size),
        #     mode='bilinear',
        #     align_corners=False
        # )

        return logits


class DINOv3(BaseModel):

    arch = "dinov3"

    def __init__(self, config: ModelConfig, device: torch.device, **kwargs):
        super().__init__(device, config, **kwargs)

        if not DINOV3_AVAILABLE:
            raise ImportError(
                "DINOv3 requires transformers>=4.56.0. Please upgrade: pip install transformers>=4.56.0"
            )

        self.config = config
        self.device = device

        # DINOv3 model configuration
        self.model_name = config.name_or_path or "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.num_classes = config.num_classes
        self.decoder_type = config.decoder_type
        self.target_size = config.target_size
        print(f"Using target_size == {self.target_size}")
        print(f"Using num_classes == {self.num_classes}")
        self.feature_layers = [-1] #getattr(config, 'feature_layers', [-4, -3, -2, -1])  # Last 4 layers

        self.dinov3_model = Dinov3Model.from_pretrained(self.model_name)
        self.image_processor = Dinov3ImageProcessor.from_pretrained(self.model_name)

        # Freeze DINOv3 backbone
        self.freeze_backbone()

        dinov3_config = self.dinov3_model.config
        embed_dim = dinov3_config.hidden_size
        embed_dims = [embed_dim] * len(self.feature_layers)

        # Create segmentation decoder
        self.segmentation_decoder = SegmentationDecoder(
            embed_dims=embed_dims,
            num_classes=self.num_classes,
            decoder_type=self.decoder_type,
            target_size=self.target_size,
        )

        self.dinov3_model.to(device)
        self.segmentation_decoder.to(device)

    def freeze_backbone(self):
        self.dinov3_model.eval()
        for param in self.dinov3_model.parameters():
            param.requires_grad = False
        print(f"DINOv3 backbone frozen with {sum(p.numel() for p in self.dinov3_model.parameters())} parameters")

    def extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        # print(images.shape)
        with torch.no_grad():  # DINOv3 is frozen
            outputs = self.dinov3_model(
                pixel_values=images,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        # print(len(hidden_states))

        # Extract features from specified layers
        features = []
        for layer_idx in self.feature_layers:
            layer_features = hidden_states[layer_idx]  # [B, 1 + N_register + N_patches, embed_dim]

            # Remove CLS token and register tokens - keep only patch tokens
            num_register_tokens = getattr(self.dinov3_model.config, 'num_register_tokens', 0)
            patch_features = layer_features[:, 1 + num_register_tokens:, :]  # Skip CLS + register tokens

            features.append(patch_features)

        return features

    def forward(self, batch: 'DataLoaderBatchDTO') -> torch.Tensor:
        images = batch.images.to(self.device)

        features = self.extract_features(images)

        logits = self.segmentation_decoder(images, features)
        return logits

    def compute_loss(self, batch: 'DataLoaderBatchDTO', logits: torch.Tensor) -> torch.Tensor:
        targets = batch.segmentation_masks.to(self.device).long()

        if targets.shape[-2:] != logits.shape[-2:]:
            print(f"[WARNING] targets.shape[-2:] != logits.shape[-2:]: {targets.shape[-2:]} != {logits.shape[-2:]}")
            targets = F.interpolate(
                targets.float().unsqueeze(1),
                size=logits.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        loss = F.cross_entropy(logits, targets, ignore_index=255)

        return loss

    def generate(self, config: GenerateImageConfig, **kwargs) -> List[Image.Image]:
        return []

    def get_trainable_parameters(self):
        """Return only the decoder parameters for training"""
        return self.segmentation_decoder.parameters()

    def save_model(self, path: str):
        """Save only the trainable decoder"""
        torch.save({
            'segmentation_decoder': self.segmentation_decoder.state_dict(),
            'config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'decoder_type': self.decoder_type,
                'target_size': self.target_size,
                'feature_layers': self.feature_layers,
            }
        }, path)

    def load_model(self, path: str):
        """Load the trained decoder"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.segmentation_decoder.load_state_dict(checkpoint['segmentation_decoder'])


# Alias for easier import
DINOv3SegmentationModel = DINOv3
