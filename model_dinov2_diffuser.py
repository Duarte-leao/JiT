import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2WithRegistersModel, AutoConfig

from util.model_util import RMSNorm
from decoder import GeneralDecoder


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedder followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embedder for class labels with reserved null token (last index)."""

    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class MLPDecoder(nn.Module):
    """
    Per-patch renderer: expands token channels, applies non-linearity + RMSNorm, and projects to RGB pixels.
    """

    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim, bias=True)
        self.act = nn.GELU()
        self.norm = RMSNorm(4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class DINOv2Diffuser(nn.Module):
    """
    Wraps a pre-trained DINOv2 ViT encoder as a diffusion backbone using token-based conditioning.
    """

    def __init__(self, input_size: int = 256, num_classes: int = 10, model_key: str = "dinov2_vitb14", decoder_config_path: str ="configs", pretrained_decoder_path='models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt', encoder_resolution: int = 224, decoder_patch_size: int = 16,**_kwargs):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.model_key = model_key
        self.encoder_resolution = encoder_resolution
        self.decoder_patch_size = decoder_patch_size

        # load pretrained backbone
        # self.backbone = torch.hub.load('facebookresearch/dinov2', model_key, pretrained=True)
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14_reg", pretrained=True)
        # self.backbone = Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-base", local_files_only=False)
        # self.backbone.layernorm.elementwise_affine = False
        # self.backbone.layernorm.weight = None
        # self.backbone.layernorm.bias = None

        self.patch_size = self._resolve_patch_size()
        assert self.patch_size[0] == self.patch_size[1], "Non-square patches are not supported."
        if self.encoder_resolution % self.patch_size[0] != 0:
             raise ValueError(f"Encoder resolution {self.encoder_resolution} must be divisible by patch size {self.patch_size[0]}")

        # 2. Check if INPUT resolution fits DECODER patch size (16)
        if self.input_size % self.decoder_patch_size != 0:
            raise ValueError(f"Input size {self.input_size} must be divisible by decoder patch size {self.decoder_patch_size}")

        # cache structural properties
        self.embed_dim = self._resolve_embed_dim()
        self.num_heads = self._resolve_num_heads()
        self.num_registers = self._resolve_num_registers()
        self.num_prefix_tokens = 1 + self.num_registers  # CLS + registers
        self.pretrain_grid_size = self._resolve_pretrain_grid()

        # conditioning modules
        self.t_embedder = TimestepEmbedder(self.embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, self.embed_dim)
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        torch.nn.init.normal_(self.cond_pos_embed, std=0.02)

        # decoder head
        # self.decoder = MLPDecoder(self.embed_dim, self.patch_size[0] * self.patch_size[1] * 3)
        self.backbone_input_size = self.encoder_resolution  
        self.backbone_patch_size = self.patch_size[0] # 14
        self.latent_dim = self.embed_dim
        assert self.backbone_input_size % self.backbone_patch_size == 0, f"backbone_input_size {self.backbone_input_size} must be divisible by backbone_patch_size {self.backbone_patch_size}"
        self.base_patches = (self.backbone_input_size // self.backbone_patch_size) ** 2 # number of patches of the latent

        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
         
        decoder_config.image_size = int(decoder_config.patch_size * torch.sqrt(torch.tensor(self.base_patches))) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        # # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.mask_head = nn.Linear(self.embed_dim, 1)

        # imagenet normalization buffers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imgnet_mean", mean)
        self.register_buffer("imgnet_std", std)

        # default curriculum state: backbone frozen, conditioning + decoder trainable
        self.freeze_backbone()
        self._ensure_trainable_cond_and_decoder()

    def _resolve_patch_size(self):
        patch_size = getattr(self.backbone.patch_embed, "patch_size", (14, 14))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        return tuple(patch_size)

    def _resolve_embed_dim(self):
        if hasattr(self.backbone, "embed_dim"):
            return self.backbone.embed_dim
        return self.backbone.pos_embed.shape[-1]

    def _resolve_num_heads(self):
        if hasattr(self.backbone, "num_heads"):
            return self.backbone.num_heads
        return self.backbone.blocks[0].attn.num_heads

    def _resolve_num_registers(self):
        if hasattr(self.backbone, "num_register_tokens"):
            return self.backbone.num_register_tokens
        if hasattr(self.backbone, "register_tokens"):
            return self.backbone.register_tokens.shape[1]
        return 0

    def _resolve_pretrain_grid(self):
        prefix_in_pos_embed = 1 
        total_tokens = self.backbone.pos_embed.shape[1]
        patch_tokens = total_tokens - prefix_in_pos_embed
        grid = int(round(patch_tokens ** 0.5))
        assert grid * grid == patch_tokens, "Patch token count is not a perfect square."
        return grid

    def _ensure_trainable_cond_and_decoder(self):
        for module in [self.t_embedder, self.y_embedder, self.cond_pos_embed, self.decoder, self.mask_head]:
            for p in module.parameters() if hasattr(module, "parameters") else [module]:
                p.requires_grad = True

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._ensure_trainable_cond_and_decoder()

    def unfreeze_last_blocks(self, n: int):
        self.freeze_backbone()
        if n <= 0:
            return
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        for block in self.backbone.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        self._ensure_trainable_cond_and_decoder()

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._ensure_trainable_cond_and_decoder()

    def _interpolate_pos_embed(self, h: int, w: int) -> torch.Tensor:
        pos_embed = self.backbone.pos_embed
        # FIX: Split at 1 (CLS), not num_prefix_tokens (CLS+Registers)
        prefix = pos_embed[:, :1] 
        patch_pos = pos_embed[:, 1:] 
        
        patch_pos = patch_pos.reshape(1, self.pretrain_grid_size, self.pretrain_grid_size, self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, self.embed_dim)
        return prefix, patch_pos

    def forward(self, x: torch.Tensor, t: torch.Tensor = None, y: torch.Tensor = None, return_features: bool = False):
        # map [-1, 1] -> [0, 1], clamp, then ImageNet normalize
        x = (x + 1) * 0.5
        x = x.clamp(0.0, 1.0)

        if x.shape[-1] != self.encoder_resolution:
            x_encoder = F.interpolate(
                x, 
                size=(self.encoder_resolution, self.encoder_resolution), 
                mode='bicubic', 
                align_corners=False
            )
        else:
            x_encoder = x
        x_encoder = (x_encoder - self.imgnet_mean.to(device=x.device, dtype=x.dtype)) / self.imgnet_std.to(
            device=x.device, dtype=x.dtype
        )

        B, _, H, W = x.shape
        assert H == self.input_size and W == self.input_size, "Input resolution mismatch."
        H_p = self.encoder_resolution // self.patch_size[0]
        W_p = self.encoder_resolution // self.patch_size[1]
        patch_tokens_per_side = H_p * W_p

        has_cond = t is not None and y is not None

        # --- CHANGED: Pass RESIZED image to backbone ---
        patch_tokens = self.backbone.patch_embed(x_encoder)
        prefix_pos, patch_pos = self._interpolate_pos_embed(H_p, W_p)
        patch_tokens = patch_tokens + patch_pos

        # backbone special tokens
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        
        # FIX: Add pos embed to CLS *before* adding registers
        cls_token = cls_token + prefix_pos 

        if self.num_registers > 0:
            register_tokens = self.backbone.register_tokens.expand(B, -1, -1)
            # Registers don't get prefix_pos added to them
            special_tokens = torch.cat([cls_token, register_tokens], dim=1)
        else:
            special_tokens = cls_token
            # If no registers, cls_token already has pos added
        
        # REMOVED: special_tokens = special_tokens + prefix_pos (Logic moved up)

        tokens_list = [special_tokens, patch_tokens]
        if has_cond:
            # conditioning tokens with dedicated positional embeddings
            t_emb = self.t_embedder(t)
            y_emb = self.y_embedder(y)
            cond_tokens = torch.stack((t_emb, y_emb), dim=1)
            cond_tokens = cond_tokens + self.cond_pos_embed
            tokens_list.insert(0, cond_tokens)

        tokens = torch.cat(tokens_list, dim=1)

        feature_idxs = {3, 7, 11}
        features = []
        for idx, block in enumerate(self.backbone.blocks):
            tokens = block(tokens)
            if return_features and idx in feature_idxs:
                # align CLS + patches regardless of conditioning/register tokens
                cls_idx = 2 if has_cond else 0
                patch_start = 2 + self.num_prefix_tokens if has_cond else self.num_prefix_tokens
                cls_token_aligned = tokens[:, cls_idx:cls_idx + 1, :]
                patch_tokens_aligned = tokens[:, patch_start:patch_start + patch_tokens_per_side, :]
                aligned = torch.cat([cls_token_aligned, patch_tokens_aligned], dim=1)
                features.append(aligned)
        tokens = self.backbone.norm(tokens)

        # decoder: drop conditioning + backbone prefix tokens; mask head must see the same slice
        prefix_len = (2 + self.num_prefix_tokens) if has_cond else self.num_prefix_tokens
        patch_tokens = tokens[:, prefix_len:, :]
        
        img_tokens = self.decoder(patch_tokens)
        mask_logits = self.mask_head(patch_tokens)

        # reshape back to image space using unpatchify
        output = self.unpatchify(img_tokens, H_p, W_p)
        if return_features:
            return output, mask_logits, features
        return output, mask_logits

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        x: (B, h*w, patch_size**2 * 3) -> (B, 3, H, W)
        """
        # --- CHANGED: Use decoder patch size ---
        p = self.decoder_patch_size # 16
        c = 3

        x = x.logits
        # Reshape: (Batch, 16, 16, 16, 16, 3)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('bhwpqc->bchpwq', x)
        # Final: (Batch, 3, 256, 256)
        x = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return x


def DINOv2_JiT_S_14(**kwargs):
    return DINOv2Diffuser(model_key="dinov2_vits14", **kwargs)


def DINOv2_JiT_B_14(**kwargs):
    return DINOv2Diffuser(model_key="dinov2_vitb14", **kwargs)
