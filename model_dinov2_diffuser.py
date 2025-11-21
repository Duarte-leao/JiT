import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.model_util import RMSNorm


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


class DINOv2Diffuser(nn.Module):
    """
    Wraps a pre-trained DINOv2 ViT encoder as a diffusion backbone using token-based conditioning.
    """

    def __init__(self, input_size: int = 224, num_classes: int = 10, model_key: str = "dinov2_vitb14", **_kwargs):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.model_key = model_key

        # load pretrained backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_key, pretrained=True)

        self.patch_size = self._resolve_patch_size()
        assert self.patch_size[0] == self.patch_size[1], "Non-square patches are not supported."
        assert self.input_size % self.patch_size[0] == 0, "Input size must be a multiple of patch size."

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
        self.decoder_norm = RMSNorm(self.embed_dim)
        self.decoder_linear = nn.Linear(self.embed_dim, self.patch_size[0] * self.patch_size[1] * 3)

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
        prefix = self.num_prefix_tokens
        total_tokens = self.backbone.pos_embed.shape[1]
        patch_tokens = total_tokens - prefix
        grid = int(round(patch_tokens ** 0.5))
        assert grid * grid == patch_tokens, "Patch token count is not a perfect square."
        return grid

    def _ensure_trainable_cond_and_decoder(self):
        for module in [self.t_embedder, self.y_embedder, self.cond_pos_embed, self.decoder_norm, self.decoder_linear]:
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
        prefix = pos_embed[:, :self.num_prefix_tokens]
        patch_pos = pos_embed[:, self.num_prefix_tokens:]
        patch_pos = patch_pos.reshape(1, self.pretrain_grid_size, self.pretrain_grid_size, self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, self.embed_dim)
        return prefix, patch_pos

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # map [-1, 1] -> [0, 1], clamp, then ImageNet normalize
        x = (x + 1) * 0.5
        x = x.clamp(0.0, 1.0)
        x = (x - self.imgnet_mean.to(device=x.device, dtype=x.dtype)) / self.imgnet_std.to(
            device=x.device, dtype=x.dtype
        )

        B, _, H, W = x.shape
        assert H == self.input_size and W == self.input_size, "Input resolution mismatch."
        H_p, W_p = H // self.patch_size[0], W // self.patch_size[1]

        # patch embed
        patch_tokens = self.backbone.patch_embed(x)  # (B, H_p*W_p, D)
        prefix_pos, patch_pos = self._interpolate_pos_embed(H_p, W_p)
        patch_tokens = patch_tokens + patch_pos

        # backbone special tokens
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        if self.num_registers > 0:
            register_tokens = self.backbone.register_tokens.expand(B, -1, -1)
            special_tokens = torch.cat([cls_token, register_tokens], dim=1)
        else:
            special_tokens = cls_token
        special_tokens = special_tokens + prefix_pos

        # conditioning tokens with dedicated positional embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        cond_tokens = torch.stack((t_emb, y_emb), dim=1)
        cond_tokens = cond_tokens + self.cond_pos_embed

        tokens = torch.cat([cond_tokens, special_tokens, patch_tokens], dim=1)

        for block in self.backbone.blocks:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)

        # decoder: drop conditioning + backbone prefix tokens
        prefix_len = 2 + self.num_prefix_tokens
        patch_tokens = tokens[:, prefix_len:, :]
        patch_tokens = self.decoder_norm(patch_tokens)
        patch_tokens = self.decoder_linear(patch_tokens)

        # reshape back to image space
        output = self.unpatchify(patch_tokens, H_p, W_p)
        return output

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        x: (B, h*w, patch_size**2 * 3) -> (B, 3, H, W)
        """
        p = self.patch_size[0]
        c = 3
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('bhwpqc->bchwqp', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))


def DINOv2_JiT_S_14(**kwargs):
    return DINOv2Diffuser(model_key="dinov2_vits14", **kwargs)


def DINOv2_JiT_B_14(**kwargs):
    return DINOv2Diffuser(model_key="dinov2_vitb14", **kwargs)
