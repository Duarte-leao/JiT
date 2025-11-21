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

    def _sample_mask(self, batch_size: int) -> torch.Tensor:
        """
        Sample a per-image grid mask with exactly K noisy patches.
        Returns a mask of shape (B, 1, grid_h, grid_w).
        """
        # one p per batch for stability (can extend to per-image if desired)
        p = torch.empty(1, device=self.device).uniform_(self.p_min, self.p_max).item()
        k = int(round(p * self.num_patches))
        k = max(0, min(k, self.num_patches))

        mask_grid = torch.zeros(batch_size, self.num_patches, device=self.device)
        if k > 0:
            idx = torch.rand(batch_size, self.num_patches, device=self.device).argsort(dim=1)[:, :k]
            # scatter 1s at selected indices
            flat_indices = idx + torch.arange(batch_size, device=self.device).unsqueeze(1) * self.num_patches
            mask_grid = mask_grid.view(-1)
            mask_grid[flat_indices.reshape(-1)] = 1.0
            mask_grid = mask_grid.view(batch_size, self.num_patches)
        mask_grid = mask_grid.view(batch_size, 1, self.grid_h, self.grid_w)
        return mask_grid

    def corrupt(self, x_start: torch.Tensor, sample_t_fn) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mosaic corruption to x_start.
        Returns:
            z_out: mosaic-composited input (B, C, H, W)
            t:     clamped timesteps (B,)
        """
        bsz = x_start.shape[0]

        # timestep sampling and clamping per stage
        t = sample_t_fn(bsz, device=x_start.device)
        t = t.clamp(max=self.t_max)
        t_view = t.view(bsz, *([1] * (x_start.ndim - 1)))

        # full noisy candidate
        e = torch.randn_like(x_start)
        z_noisy = t_view * x_start + (1 - t_view) * e

        # patch mask generation on grid and upsample to pixel space
        mask_grid = self._sample_mask(bsz)  # (B,1,grid_h,grid_w)
        mask_pixel = mask_grid.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)

        # compose mosaic
        z_out = x_start * (1 - mask_pixel) + z_noisy * mask_pixel
        return z_out, t


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
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

        # curriculum stages: (end_epoch, stage_cfg, backbone_action)
        # Defaults target ImageNette validation; adjust epochs as needed.
        self.stage_schedule = [
            (args.epochs // 3, {"p_min": 0.10, "p_max": 0.30, "t_max": 0.3}, "freeze"),
            (2 * args.epochs // 3, {"p_min": 0.40, "p_max": 0.70, "t_max": 0.6}, "unfreeze_last"),
            (args.epochs + 1, {"p_min": 0.80, "p_max": 1.00, "t_max": 1.0}, "unfreeze_all"),
        ]
        self.current_stage_idx = -1
        # initialize stage 0
        self.set_epoch(args.start_epoch)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def set_epoch(self, epoch: int):
        """Update curriculum stage and backbone freezing according to current epoch."""
        # find first schedule entry whose end_epoch is greater than epoch
        for idx, (end_epoch, stage_cfg, action) in enumerate(self.stage_schedule):
            if epoch < end_epoch:
                # only react on stage change
                if idx != self.current_stage_idx:
                    self.current_stage_idx = idx
                    self.mosaic_engine.update_stage(stage_cfg)
                    if hasattr(self.net, "freeze_backbone"):
                        if action == "freeze":
                            self.net.freeze_backbone()
                        elif action == "unfreeze_last":
                            # default to last 4 blocks; adjust easily if needed
                            self.net.unfreeze_last_blocks(4)
                        elif action == "unfreeze_all":
                            self.net.unfreeze_all()
                break

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        z, t = self.mosaic_engine.corrupt(x, self.sample_t)
        t_view = t.view(-1, *([1] * (x.ndim - 1)))

        v = (x - z) / (1 - t_view).clamp_min(self.t_eps)

        x_pred = self.net(z, t, labels_dropped)
        v_pred = (x_pred - z) / (1 - t_view).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

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
