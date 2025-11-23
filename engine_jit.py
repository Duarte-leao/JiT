import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy
import torchvision.utils as vutils
from typing import Optional
import math as pymath


def _upsample_patch_mask(mask_patch_logits: Optional[torch.Tensor], patch_size: int, H: int, W: int) -> Optional[torch.Tensor]:
    """
    Convert patch-level mask logits (B, N, 1) to pixel space (B, 1, H, W).
    """
    if mask_patch_logits is None:
        return None
    B = mask_patch_logits.shape[0]
    H_p, W_p = H // patch_size, W // patch_size
    mask_grid = mask_patch_logits.permute(0, 2, 1).reshape(B, 1, H_p, W_p)
    mask_pixel = mask_grid.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return mask_pixel


def _apply_gating(x_pred: torch.Tensor, mask_patch_logits: Optional[torch.Tensor], z_input: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Apply self-gating using patch-level mask logits; fall back to raw prediction if mask is unavailable.
    """
    if mask_patch_logits is None:
        return x_pred
    mask_pixel = _upsample_patch_mask(mask_patch_logits, patch_size, x_pred.shape[2], x_pred.shape[3])
    if mask_pixel is None:
        return x_pred
    gate = torch.sigmoid(mask_pixel.to(dtype=x_pred.dtype)/0.2)
    return gate * x_pred + (1 - gate) * z_input


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('stage', misc.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('x_mse', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mask_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # synchronize curriculum stage with epoch
    if hasattr(model_without_ddp, "set_epoch"):
        model_without_ddp.set_epoch(epoch)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # static curriculum state for this epoch (used for logging/printing)
    curriculum_state = None
    if hasattr(model_without_ddp, "get_curriculum_state"):
        curriculum_state = model_without_ddp.get_curriculum_state()
        if curriculum_state:
            print(
                f"Curriculum: stage {curriculum_state.get('stage', 0)}, "
                f"p_max={curriculum_state.get('p_max', 0.0):.2f}, "
                f"t_max={curriculum_state.get('t_max', 0.0):.2f}, "
                f"backbone={curriculum_state.get('backbone_mode', 'n/a')}"
            )

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss_out = model(x, labels)
            loss = x_mse = mask_loss = None
            if isinstance(loss_out, tuple):
                if len(loss_out) == 3:
                    loss, x_mse, mask_loss = loss_out
                elif len(loss_out) == 2:
                    loss, x_mse = loss_out
                else:
                    loss = loss_out[0]
            else:
                loss = loss_out

        loss_value = loss.item()
        x_mse_value = x_mse.item() if x_mse is not None else None
        mask_loss_value = mask_loss.item() if mask_loss is not None else None
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if curriculum_state:
            metric_logger.update(stage=int(curriculum_state.get("stage", 0)))
        if x_mse_value is not None:
            metric_logger.update(x_mse=x_mse_value)
        if mask_loss_value is not None:
            metric_logger.update(mask_loss=mask_loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        x_mse_reduce = misc.all_reduce_mean(x_mse_value) if x_mse_value is not None else None
        mask_loss_reduce = misc.all_reduce_mean(mask_loss_value) if mask_loss_value is not None else None

        if log_writer is not None:
            # Use a monotonically increasing step across epochs/iters
            global_step = data_iter_step + epoch * args.steps_per_epoch
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, global_step)
                log_writer.add_scalar('lr', lr, global_step)
                if x_mse_reduce is not None:
                    log_writer.add_scalar('train/x_mse', x_mse_reduce, global_step)
                if mask_loss_reduce is not None:
                    log_writer.add_scalar('train/mask_loss', mask_loss_reduce, global_step)
                # curriculum metrics
                if curriculum_state is not None:
                    log_writer.add_scalar('curriculum/stage', curriculum_state.get("stage", 0), global_step)
                    if "p_max" in curriculum_state:
                        log_writer.add_scalar('curriculum/p_max', curriculum_state["p_max"], global_step)
                    if "t_max" in curriculum_state:
                        log_writer.add_scalar('curriculum/t_max', curriculum_state["t_max"], global_step)
                    if "backbone_state_code" in curriculum_state:
                        log_writer.add_scalar('curriculum/backbone_state_code', curriculum_state["backbone_state_code"], global_step)
                # wandb logging (main process only)
                if getattr(args, "use_wandb", False) and misc.is_main_process():
                    try:
                        import wandb
                        log_payload = {'train/loss': loss_value_reduce, 'train/lr': lr, 'train/epoch': epoch}
                        if x_mse_reduce is not None:
                            log_payload['train/x_mse'] = x_mse_reduce
                        if mask_loss_reduce is not None:
                            log_payload['train/mask_loss'] = mask_loss_reduce
                        if curriculum_state is not None:
                            log_payload.update({
                                'curriculum/stage': curriculum_state.get("stage", 0),
                                'curriculum/p_max': curriculum_state.get("p_max", 0.0),
                                'curriculum/t_max': curriculum_state.get("t_max", 0.0),
                                'curriculum/backbone_state_code': curriculum_state.get("backbone_state_code", 0),
                                'curriculum/backbone_mode': curriculum_state.get("backbone_mode", ""),
                            })
                        wandb.log(log_payload, step=global_step)
                    except Exception as e:
                        # keep training even if wandb log fails
                        print(f"W&B log warning: {e}")


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        "ssd/tmp",
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()


@torch.no_grad()
def _optional_lpips():
    try:
        import lpips
        return lpips.LPIPS(net='vgg')
    except Exception as e:
        print(f"LPIPS not available ({e}); skipping LPIPS computation.")
        return None


@torch.no_grad()
def run_restoration_eval(model_without_ddp, data_loader_val, device, epoch, args):
    """
    Stage-aware restoration evaluator: deterministic mosaic corruption, single-step prediction,
    PSNR/LPIPS metrics, and triplet grid saving.
    """
    if data_loader_val is None or args.restoration_eval_freq <= 0 or args.restoration_eval_num <= 0:
        return

    # read curriculum stage
    curriculum_state = model_without_ddp.get_curriculum_state() if hasattr(model_without_ddp, "get_curriculum_state") else None
    if curriculum_state is None:
        return
    stage = curriculum_state.get("stage", 0)
    if stage < 1:
        return

    # switch to EMA params
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    model_without_ddp.load_state_dict(ema_state_dict)

    was_training = model_without_ddp.training
    model_without_ddp.eval()

    # collect fixed subset
    imgs = []
    labels = []
    it = iter(data_loader_val)
    while len(imgs) * data_loader_val.batch_size < args.restoration_eval_num:
        try:
            batch_imgs, batch_labels = next(it)
        except StopIteration:
            break
        imgs.append(batch_imgs)
        labels.append(batch_labels)
    if len(imgs) == 0:
        model_without_ddp.load_state_dict(model_state_dict)
        if was_training:
            model_without_ddp.train()
        return

    x = torch.cat(imgs, dim=0)[:args.restoration_eval_num].to(device, non_blocking=True).float()
    if x.max() > 1.0:
        x.div_(255)
    y = torch.cat(labels, dim=0)[:args.restoration_eval_num].to(device, non_blocking=True)
    x = x * 2.0 - 1.0  # [-1,1]

    # deterministic mosaic corruption
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        z_mosaic, t, _ = model_without_ddp.mosaic_engine.corrupt(x, model_without_ddp.sample_t)

    # forward (no label drop, cfg=1)
    net_out = model_without_ddp.net(z_mosaic, t, y)
    if isinstance(net_out, tuple):
        x_pred, mask_logits_patch = net_out
    else:
        x_pred, mask_logits_patch = net_out, None
    patch_size = model_without_ddp.mosaic_engine.patch_size
    x_pred_gated = _apply_gating(x_pred, mask_logits_patch, z_mosaic, patch_size)

    def denorm(tensor):
        return torch.clamp((tensor + 1) / 2, 0.0, 1.0)

    # metrics in image space [0,1]
    x_gt = denorm(x)
    x_pd = denorm(x_pred_gated)
    mse = (x_gt - x_pd).pow(2).flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    psnr_mean = psnr.mean().item()

    lpips_loss_fn = _optional_lpips()
    lpips_val = None
    if lpips_loss_fn is not None:
        lpips_loss_fn = lpips_loss_fn.to(device)
        # LPIPS expects [-1,1]
        lpips_vals = lpips_loss_fn(x, x_pred_gated).flatten()
        lpips_val = lpips_vals.mean().item()

    # grids
    x_clean = x_gt.cpu()
    x_input = denorm(z_mosaic).cpu()
    x_out = x_pd.cpu()
    panels = []
    for i in range(x_clean.size(0)):
        panel = torch.cat([x_clean[i], x_input[i], x_out[i]], dim=2)
        panels.append(panel)
        if len(panels) >= 32:  # keep grid manageable
            break
    grid = vutils.make_grid(panels, nrow=int(len(panels) ** 0.5) or 1)

    save_dir = os.path.join(args.output_dir, "val_restoration", f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "restoration.png")
    vutils.save_image(grid, save_path)

    # logging
    if log_writer := getattr(run_restoration_eval, "_log_writer", None):
        log_writer.add_scalar('val/restoration_psnr', psnr_mean, epoch)
        if lpips_val is not None:
            log_writer.add_scalar('val/restoration_lpips', lpips_val, epoch)

    if getattr(args, "use_wandb", False) and misc.is_main_process():
        try:
            import wandb
            eval_step = (epoch + 1) * args.steps_per_epoch
            log_payload = {'val/restoration_psnr': psnr_mean}
            if lpips_val is not None:
                log_payload['val/restoration_lpips'] = lpips_val
            wandb.log(log_payload, step=eval_step)
            wandb.log({"val/restoration_grid": wandb.Image(save_path, caption=f"epoch {epoch}, stage {stage}")}, step=eval_step)
        except Exception as e:
            print(f"W&B restoration log warning: {e}")

    # restore state
    model_without_ddp.load_state_dict(model_state_dict)
    if was_training:
        model_without_ddp.train()


@torch.no_grad()
def run_multistep_restoration(model_without_ddp, data_loader_val, device, epoch, args):
    """
    Multi-step restoration visualizer: starts from fixed-t mosaic, integrates t_max->0 with ODE steps,
    compares single-step vs multi-step outputs, and logs PSNR/LPIPS plus a quad grid.
    """
    if data_loader_val is None or args.recons_multistep_freq <= 0 or args.restoration_eval_num <= 0:
        return

    # curriculum state for t_max
    curriculum_state = model_without_ddp.get_curriculum_state() if hasattr(model_without_ddp, "get_curriculum_state") else None
    if curriculum_state is None:
        return
    t_max = float(curriculum_state.get("t_max", 1.0))
    start_t = 1.0 - t_max  # curriculum t_max is max noise, so signal starts at 1 - t_max

    # swap to EMA params
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    model_without_ddp.load_state_dict(ema_state_dict)

    was_training = model_without_ddp.training
    model_without_ddp.eval()

    # fixed subset
    imgs = []
    labels = []
    it = iter(data_loader_val)
    while len(imgs) * data_loader_val.batch_size < args.restoration_eval_num:
        try:
            batch_imgs, batch_labels = next(it)
        except StopIteration:
            break
        imgs.append(batch_imgs)
        labels.append(batch_labels)
    if len(imgs) == 0:
        model_without_ddp.load_state_dict(model_state_dict)
        if was_training:
            model_without_ddp.train()
        return

    x = torch.cat(imgs, dim=0)[:args.restoration_eval_num].to(device, non_blocking=True).float()
    if x.max() > 1.0:
        x.div_(255)
    y = torch.cat(labels, dim=0)[:args.restoration_eval_num].to(device, non_blocking=True)
    x = x * 2.0 - 1.0  # [-1,1]

    # deterministic mosaic at fixed t_max
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        z_mosaic, t_clamped, _ = model_without_ddp.mosaic_engine.corrupt(x, model_without_ddp.sample_t, fixed_t=start_t)

    # single-step prediction
    net_out_single = model_without_ddp.net(z_mosaic, t_clamped, y)
    if isinstance(net_out_single, tuple):
        x_pred_single, mask_logits_init = net_out_single
    else:
        x_pred_single, mask_logits_init = net_out_single, None
    patch_size = model_without_ddp.mosaic_engine.patch_size
    x_final_single = _apply_gating(x_pred_single, mask_logits_init, z_mosaic, patch_size)

    # partial trajectory integration from t_max -> 0
    timesteps = torch.linspace(start_t, 1.0, args.num_sampling_steps + 1, device=device, dtype=x.dtype)
    z_cur = z_mosaic
    def _forward_flow(z, t_scalar):
        t_view = t_scalar.view(-1, *([1] * (z.ndim - 1)))
        net_out = model_without_ddp.net(z, t_scalar, y)
        x_pred = net_out[0] if isinstance(net_out, tuple) else net_out
        v_pred = (x_pred - z) / (1.0 - t_view).clamp_min(model_without_ddp.t_eps)
        return v_pred

    for i in range(len(timesteps) - 1):
        t = timesteps[i].expand(x.shape[0])
        t_next = timesteps[i + 1].expand(x.shape[0])
        if model_without_ddp.method == "heun":
            v_pred_t = _forward_flow(z_cur, t)
            z_euler = z_cur + (t_next - t).view(-1, *([1] * (z_cur.ndim - 1))) * v_pred_t
            v_pred_t_next = _forward_flow(z_euler, t_next)
            v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        else:
            v_pred = _forward_flow(z_cur, t)
        z_cur = z_cur + (t_next - t).view(-1, *([1] * (z_cur.ndim - 1))) * v_pred

    x_pred_multi = z_cur

    def denorm(tensor):
        return torch.clamp((tensor + 1) / 2, 0.0, 1.0)

    x_gt = denorm(x)
    x_mosaic = denorm(z_mosaic)
    x_single = denorm(x_final_single)
    # gate final state once using the initial mask prediction
    x_multi = denorm(_apply_gating(x_pred_multi, mask_logits_init, z_mosaic, patch_size))

    # metrics: PSNR/LPIPS for single and multi
    def _psnr(a, b):
        mse = (a - b).pow(2).flatten(1).mean(dim=1)
        return (-10.0 * torch.log10(mse.clamp_min(1e-10))).mean().item()

    psnr_single = _psnr(x_gt, x_single)
    psnr_multi = _psnr(x_gt, x_multi)

    lpips_loss_fn = _optional_lpips()
    lpips_single = lpips_multi = None
    if lpips_loss_fn is not None:
        lpips_loss_fn = lpips_loss_fn.to(device)
        lpips_single = lpips_loss_fn(x, x_final_single).flatten().mean().item()
        lpips_multi = lpips_loss_fn(x, _apply_gating(x_pred_multi, mask_logits_init, z_mosaic, patch_size)).flatten().mean().item()

    # grids (limit to 32 panels)
    panels = []
    for i in range(x_gt.size(0)):
        panel = torch.cat([x_gt[i], x_mosaic[i], x_single[i], x_multi[i]], dim=2)
        panels.append(panel)
        if len(panels) >= 32:
            break
    grid = vutils.make_grid(panels, nrow=int(len(panels) ** 0.5) or 1)

    save_dir = os.path.join(args.output_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_multistep.png")
    vutils.save_image(grid, save_path)

    # logging
    if log_writer := getattr(run_multistep_restoration, "_log_writer", None):
        log_writer.add_scalar('val/multistep_psnr', psnr_multi, epoch)
        if lpips_multi is not None:
            log_writer.add_scalar('val/multistep_lpips', lpips_multi, epoch)

    if getattr(args, "use_wandb", False) and misc.is_main_process():
        try:
            import wandb
            eval_step = (epoch + 1) * args.steps_per_epoch
            log_payload = {
                'val/multistep_psnr': psnr_multi,
                'val/multistep_psnr_single': psnr_single,
            }
            if lpips_multi is not None:
                log_payload['val/multistep_lpips'] = lpips_multi
                log_payload['val/multistep_lpips_single'] = lpips_single
            wandb.log(log_payload, step=eval_step)
            wandb.log({"val/multistep_grid": wandb.Image(save_path, caption=f"epoch {epoch}, t_max={t_max}")}, step=eval_step)
        except Exception as e:
            print(f"W&B multistep log warning: {e}")

    # restore state
    model_without_ddp.load_state_dict(model_state_dict)
    if was_training:
        model_without_ddp.train()


@torch.no_grad()
def run_reconstructions(model_without_ddp, data_loader_val, device, epoch, args):
    """
    Generate reconstruction grids from validation images for qualitative tracking.
    Uses EMA weights, ground-truth labels, and cfg_scale=1.0 (no CFG).
    """
    if data_loader_val is None or args.num_recons <= 0:
        return

    # swap to EMA weights (first EMA) like evaluate
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    model_without_ddp.load_state_dict(ema_state_dict)

    was_training = model_without_ddp.training
    model_without_ddp.eval()

    imgs = []
    labels = []
    it = iter(data_loader_val)
    while len(imgs) < args.num_recons:
        try:
            batch_imgs, batch_labels = next(it)
        except StopIteration:
            break
        imgs.append(batch_imgs)
        labels.append(batch_labels)
    if len(imgs) == 0:
        # restore and exit
        model_without_ddp.load_state_dict(model_state_dict)
        if was_training:
            model_without_ddp.train()
        return
    x = torch.cat(imgs, dim=0)[:args.num_recons].to(device, non_blocking=True).float()
    if x.max() > 1.0:
        x.div_(255)
    y = torch.cat(labels, dim=0)[:args.num_recons].to(device, non_blocking=True)
    x = x * 2.0 - 1.0  # [-1,1]

    # Mosaic corruption with current stage
    z_mosaic, t, _ = model_without_ddp.mosaic_engine.corrupt(x, model_without_ddp.sample_t)

    # Forward pass (no label drop, cfg=1)
    net_out = model_without_ddp.net(z_mosaic, t, y)
    if isinstance(net_out, tuple):
        x_pred, mask_logits_patch = net_out
    else:
        x_pred, mask_logits_patch = net_out, None
    patch_size = model_without_ddp.mosaic_engine.patch_size
    x_pred = _apply_gating(x_pred, mask_logits_patch, z_mosaic, patch_size)

    def denorm(tensor):
        return torch.clamp((tensor + 1) / 2, 0.0, 1.0)

    x_clean = denorm(x).cpu()
    x_input = denorm(z_mosaic).cpu()
    x_out = denorm(x_pred).cpu()

    panels = []
    for i in range(x_clean.size(0)):
        panel = torch.cat([x_clean[i], x_input[i], x_out[i]], dim=2)
        panels.append(panel)
    grid = vutils.make_grid(panels, nrow=int(len(panels) ** 0.5) or 1)

    save_dir = os.path.join(args.output_dir, "images", f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "recon.png")
    vutils.save_image(grid, save_path)

    # wandb logging
    if getattr(args, "use_wandb", False) and misc.is_main_process():
        try:
            import wandb
            eval_step = (epoch + 1) * args.steps_per_epoch
            stage_info = None
            if hasattr(model_without_ddp, "get_curriculum_state"):
                stage_info = model_without_ddp.get_curriculum_state()
            caption = f"epoch {epoch}"
            if stage_info:
                caption += f", stage {stage_info.get('stage', '')}"
            wandb.log({"reconstructions": wandb.Image(save_path, caption=caption)}, step=eval_step)
        except Exception as e:
            print(f"W&B image log warning: {e}")

    # restore state
    model_without_ddp.load_state_dict(model_state_dict)
    if was_training:
        model_without_ddp.train()


@torch.no_grad()
def run_clean_reconstruction(model_without_ddp, data_loader_val, device, epoch, args):
    """
    Clean reconstruction visualizer: tests identity capability at t=0 without mosaic corruption.
    Uses EMA weights, ground-truth labels, and logs PSNR/LPIPS plus a two-row grid.
    """
    if data_loader_val is None or args.num_recons <= 0:
        return

    # swap to EMA weights (first EMA) like evaluate
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    model_without_ddp.load_state_dict(ema_state_dict)

    was_training = model_without_ddp.training
    model_without_ddp.eval()

    # fixed subset
    imgs = []
    labels = []
    it = iter(data_loader_val)
    while len(imgs) < args.num_recons:
        try:
            batch_imgs, batch_labels = next(it)
        except StopIteration:
            break
        imgs.append(batch_imgs)
        labels.append(batch_labels)
    if len(imgs) == 0:
        model_without_ddp.load_state_dict(model_state_dict)
        if was_training:
            model_without_ddp.train()
        return
    x = torch.cat(imgs, dim=0)[:args.num_recons].to(device, non_blocking=True).float()
    if x.max() > 1.0:
        x.div_(255)
    y = torch.cat(labels, dim=0)[:args.num_recons].to(device, non_blocking=True)
    x = x * 2.0 - 1.0  # [-1,1]

    # pure-signal timestep (t=1.0 corresponds to clean data in the signal-strength convention)
    t = torch.ones(x.size(0), device=device, dtype=x.dtype)

    # forward pass (no label drop, cfg=1)
    net_out = model_without_ddp.net(x, t, y)
    if isinstance(net_out, tuple):
        x_rec, mask_logits_patch = net_out
    else:
        x_rec, mask_logits_patch = net_out, None
    patch_size = model_without_ddp.mosaic_engine.patch_size
    # x_rec = _apply_gating(x_rec, mask_logits_patch, x, patch_size)

    def denorm(tensor):
        return torch.clamp((tensor + 1) / 2, 0.0, 1.0)

    x_gt = denorm(x)
    x_pd = denorm(x_rec)

    # metrics
    mse = (x_gt - x_pd).pow(2).flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    psnr_mean = psnr.mean().item()

    lpips_loss_fn = _optional_lpips()
    lpips_val = None
    if lpips_loss_fn is not None:
        lpips_loss_fn = lpips_loss_fn.to(device)
        lpips_vals = lpips_loss_fn(x, x_rec).flatten()
        lpips_val = lpips_vals.mean().item()

    # grid: two-row (clean on top, recon on bottom)
    panels = []
    for i in range(x_gt.size(0)):
        panel = torch.cat([x_gt[i], x_pd[i]], dim=1)  # stack vertically
        panels.append(panel)
        if len(panels) >= 32:
            break
    grid = vutils.make_grid(panels, nrow=int(len(panels) ** 0.5) or 1)

    save_dir = os.path.join(args.output_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_clean.png")
    vutils.save_image(grid, save_path)

    # wandb logging
    if getattr(args, "use_wandb", False) and misc.is_main_process():
        try:
            import wandb
            eval_step = (epoch + 1) * args.steps_per_epoch
            log_payload = {'val/clean_reconstruction_psnr': psnr_mean}
            if lpips_val is not None:
                log_payload['val/clean_reconstruction_lpips'] = lpips_val
            caption = f"epoch {epoch}, clean reconstruction"
            wandb.log(log_payload, step=eval_step)
            wandb.log({"validation/clean_reconstruction": wandb.Image(save_path, caption=caption)}, step=eval_step)
        except Exception as e:
            print(f"W&B clean recon log warning: {e}")

    if log_writer := getattr(run_clean_reconstruction, "_log_writer", None):
        log_writer.add_scalar('val/clean_reconstruction_psnr', psnr_mean, (epoch + 1) * args.steps_per_epoch)
        if lpips_val is not None:
            log_writer.add_scalar('val/clean_reconstruction_lpips', lpips_val, (epoch + 1) * args.steps_per_epoch)

    # restore state
    model_without_ddp.load_state_dict(model_state_dict)
    if was_training:
        model_without_ddp.train()
