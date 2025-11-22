import torch
import torch.nn as nn
from model_jit import JiT_models
from model_dinov2_diffuser import DINOv2_JiT_S_14, DINOv2_JiT_B_14

# extend registry with DINOv2 variants
JiT_models.update({
    'DINOv2-JiT-S/14': DINOv2_JiT_S_14,
    'DINOv2-JiT-B/14': DINOv2_JiT_B_14,
})


class MosaicNoisingEngine:
    """
    Builds mosaic-corrupted inputs by mixing clean and noisy patches according to a curriculum.
    """
    def __init__(self, img_size: int, patch_size: int, device: torch.device):
        self.img_size = img_size
        self.patch_size = patch_size
        self.device = device

        assert self.img_size % self.patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_h = self.img_size // self.patch_size
        self.grid_w = self.img_size // self.patch_size
        self.num_patches = self.grid_h * self.grid_w

        # stage parameters (updated via update_stage)
        self.p_min = 0.1
        self.p_max = 0.3
        self.t_max = 1.0

    def update_stage(self, stage_cfg: dict):
        """Update the corruption intensity for the active stage."""
        self.p_min = stage_cfg.get("p_min", self.p_min)
        self.p_max = stage_cfg.get("p_max", self.p_max)
        self.t_max = stage_cfg.get("t_max", self.t_max)

    def _sample_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample a per-image grid mask with exactly K noisy patches.
        Returns a mask of shape (B, 1, grid_h, grid_w).
        """
        # per-image noise ratios
        p = torch.rand(batch_size, device=device) * (self.p_max - self.p_min) + self.p_min
        k = (p * self.num_patches).round().long().clamp_(0, self.num_patches)

        noise = torch.rand(batch_size, self.num_patches, device=device)
        noise_sorted, _ = noise.sort(dim=1)
        idx = (self.num_patches - k).clamp(min=0, max=self.num_patches - 1).unsqueeze(1)
        thresholds = torch.where(
            k.unsqueeze(1) > 0,
            noise_sorted.gather(1, idx),
            torch.ones(batch_size, 1, device=device) * 2.0  # > 1 to force zeros when k=0
        )
        mask_flat = (noise >= thresholds).float()
        mask_grid = mask_flat.view(batch_size, 1, self.grid_h, self.grid_w)
        return mask_grid

    def corrupt(self, x_start: torch.Tensor, sample_t_fn, fixed_t: float = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mosaic corruption to x_start.
        Returns:
            z_out: mosaic-composited input (B, C, H, W)
            t:     clamped timesteps (B,)
        """
        bsz = x_start.shape[0]
        device = x_start.device

        # timestep sampling and clamping per stage (or fixed)
        if fixed_t is not None:
            t = torch.full((bsz,), float(fixed_t), device=device, dtype=x_start.dtype)
        else:
            t = sample_t_fn(bsz, device=device)
            # curriculum t_max refers to max noise => min signal = 1 - t_max
            t = t.clamp(min=1.0 - self.t_max)
        t_view = t.view(bsz, *([1] * (x_start.ndim - 1)))

        # full noisy candidate
        e = torch.randn_like(x_start)
        z_noisy = t_view * x_start + (1 - t_view) * e

        # patch mask generation on grid and upsample to pixel space
        mask_grid = self._sample_mask(bsz, device=device)  # (B,1,grid_h,grid_w)
        mask_pixel = mask_grid.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)

        # compose mosaic
        z_out = x_start * (1 - mask_pixel) + z_noisy * mask_pixel
        return z_out, t


class CurriculumController:
    """
    Coordinates curriculum stages across data corruption difficulty and backbone plasticity.
    """
    def __init__(self, total_epochs: int, mosaic_engine: MosaicNoisingEngine, backbone: nn.Module):
        self.total_epochs = total_epochs
        self.mosaic_engine = mosaic_engine
        self.backbone = backbone
        self.current_stage_idx = -1
        self.backbone_state_code = 2  # default unfreeze
        self.backbone_mode_str = "Unfrozen"

        # Stage definitions using fractional start points
        self.stages = [
            {"start_frac": 0.0, "p_min": 0.10, "p_max": 0.30, "t_max": 0.3, "backbone_mode": "freeze"},
            {"start_frac": 0.3, "p_min": 0.40, "p_max": 0.70, "t_max": 0.6, "backbone_mode": "unfreeze_last_4"},
            {"start_frac": 0.7, "p_min": 0.80, "p_max": 1.00, "t_max": 1.0, "backbone_mode": "unfreeze_all"},
        ]
        # derive robust start epochs ensuring stage 1 starts at 0 and subsequent stages do not collapse
        prev_start = 0
        for i, stage in enumerate(self.stages):
            if i == 0:
                stage["start_epoch"] = 0
                prev_start = 0
                continue
            start_ep = int(stage["start_frac"] * self.total_epochs)
            start_ep = max(prev_start + 1, start_ep if start_ep > 0 else 1)
            stage["start_epoch"] = start_ep
            prev_start = start_ep
        self.stages = sorted(self.stages, key=lambda s: s["start_epoch"])

    def _resolve_stage_idx(self, epoch: int) -> int:
        """
        Pick the stage with the largest start_epoch <= current epoch.
        """
        idx = 0
        for i, stage in enumerate(self.stages):
            if epoch >= stage["start_epoch"]:
                idx = i
            else:
                break
        return idx

    def _apply_backbone_mode(self, mode: str):
        """
        Execute backbone freezing policy if available; record state code and label.
        """
        is_dino = hasattr(self.backbone, "freeze_backbone")
        if mode == "freeze":
            if is_dino:
                self.backbone.freeze_backbone()
            self.backbone_state_code = 0
            self.backbone_mode_str = "Frozen"
        elif mode == "unfreeze_last_4":
            if is_dino:
                self.backbone.unfreeze_last_blocks(4)
            self.backbone_state_code = 1
            self.backbone_mode_str = "Partial"
        elif mode == "unfreeze_all":
            if is_dino:
                self.backbone.unfreeze_all()
            self.backbone_state_code = 2
            self.backbone_mode_str = "Unfrozen"
        else:
            # Unknown mode: fallback to no-op, mark as partial
            self.backbone_state_code = 1
            self.backbone_mode_str = mode

    def set_epoch(self, epoch: int):
        """
        Update active stage based on epoch. Applies both data and backbone configs.
        """
        stage_idx = self._resolve_stage_idx(epoch)
        if stage_idx == self.current_stage_idx:
            return self.get_curriculum_state()

        self.current_stage_idx = stage_idx
        stage_cfg = self.stages[stage_idx]

        # Push data difficulty
        if self.mosaic_engine is not None:
            self.mosaic_engine.update_stage(stage_cfg)

        # Apply backbone mode (no-op for JiT)
        self._apply_backbone_mode(stage_cfg.get("backbone_mode", "unfreeze_all"))

        return self.get_curriculum_state()

    def get_curriculum_state(self):
        if self.current_stage_idx < 0 or self.current_stage_idx >= len(self.stages):
            return None
        stage_cfg = self.stages[self.current_stage_idx]
        return {
            "stage": self.current_stage_idx + 1,
            "p_min": stage_cfg.get("p_min", 0.0),
            "p_max": stage_cfg.get("p_max", 0.0),
            "t_max": stage_cfg.get("t_max", 0.0),
            "backbone_mode": self.backbone_mode_str,
            "backbone_state_code": self.backbone_state_code,
        }


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        # enforce patch-size compatibility for DINOv2 backbones
        if args.model in ['DINOv2-JiT-S/14', 'DINOv2-JiT-B/14'] and args.img_size % 14 != 0:
            raise ValueError(
                f"Model {args.model} requires img_size to be a multiple of 14 (e.g., 224, 280). "
                f"Got img_size={args.img_size}."
            )
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # mosaic noising engine (uses patch geometry from the backbone)
        patch_size = self.net.patch_size if isinstance(self.net.patch_size, int) else self.net.patch_size[0]
        self.mosaic_engine = MosaicNoisingEngine(img_size=self.img_size, patch_size=patch_size, device=torch.device(args.device))
        # curriculum controller
        self.curriculum = CurriculumController(total_epochs=args.epochs, mosaic_engine=self.mosaic_engine, backbone=self.net)
        self.curriculum.set_epoch(args.start_epoch)

    def get_curriculum_state(self):
        """Expose current curriculum state (stage + key parameters)."""
        return self.curriculum.get_curriculum_state()

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def set_epoch(self, epoch: int):
        """Delegate curriculum update to the controller."""
        return self.curriculum.set_epoch(epoch)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _compute_losses(self, x, z, t, x_pred):
        """
        Compute Flow Matching loss (v-space) and x-space MSE.
        All tensors are expected to be in image space [-1, 1] with shape (B, C, H, W).
        """
        t_view = t.view(-1, *([1] * (x.ndim - 1)))

        # Flow Matching targets/preds
        v_target = (x - z) / (1 - t_view).clamp_min(self.t_eps)
        v_pred = (x_pred - z) / (1 - t_view).clamp_min(self.t_eps)

        loss_v = (v_target - v_pred) ** 2
        loss_v = loss_v.mean(dim=(1, 2, 3)).mean()

        # x-space diagnostic
        x_mse = ((x - x_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        return loss_v, x_mse

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        z, t = self.mosaic_engine.corrupt(x, self.sample_t)
        x_pred = self.net(z, t, labels_dropped)

        loss_v, x_mse = self._compute_losses(x, z, t, x_pred)
        return loss_v, x_mse

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
