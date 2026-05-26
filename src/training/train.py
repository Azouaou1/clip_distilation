import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, KDClipLoss, get_cast_dtype
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


import matplotlib
matplotlib.use('Agg')  # pas de display, sauvegarde PNG uniquement
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
import numpy as np
import os
import torch.nn.functional as F



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch2(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    # --- INITIALISATION STRATÉGIQUE POUR SIGLIP ---
    # Au tout début du fine-tuning, on force le scale à démarrer plus bas (ln(10))
    # pour éviter que la sigmoïde ne sature immédiatement à cause du checkpoint pré-entraîné
    if epoch == 0:
        with torch.no_grad():
            unwrap_model(model).logit_scale.copy_(torch.tensor(math.log(10.0)))
            if hasattr(unwrap_model(model), 'logit_bias'):
                unwrap_model(model).logit_bias.copy_(torch.tensor(-10.0))

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            # 1. Extraction des features via le forward du modèle
            image_features, text_features, logit_scale = model(images, texts)

            # 2. Récupération de la valeur brute moyenne du scale
            current_scale = logit_scale.mean()

            # 3. Récupération du biais apprenable s'il existe dans l'architecture
            model_unwrapped = unwrap_model(model)
            bias_val = model_unwrapped.logit_bias if hasattr(model_unwrapped, 'logit_bias') else 0.0

            # 4. Calcul de la matrice de similarité (produit scalaire)
            similarity = image_features @ text_features.t()

            # 5. Application de la transformation linéaire SigLIP (scale * sim + bias)
            logits_per_image = (current_scale * similarity) + bias_val

            # 6. Construction de la matrice cible (+1 pour la diagonale, -1 pour le reste)
            batch_size = images.shape[0]
            #labels = torch.eye(batch_size, device=device) * 2.0 - 1.0
            eps = 0.1
            labels = torch.eye(batch_size, device=device) * (2.0 - 2 * eps) - (1.0 - 2 * eps)

            # 7. Perte logsigmoïde binaire stable paire par paire
            total_loss = -F.logsigmoid(logits_per_image * labels).sum() / batch_size

        # --- RETOUR AU PIPELINE DE BACKWARD STANDARD ---
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
             unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.mean().item()

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss (SigLIP): {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size * args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(10))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size * args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def train_kd_one_epoch(model, t_model, data, epoch, loss, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_task = AverageMeter()
    loss_icl = AverageMeter()
    loss_ckd = AverageMeter()
    loss_cross_kd = AverageMeter()
    loss_fd = AverageMeter()
    loss_gd = AverageMeter()
    loss_afd = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts, distill=True, mask_ratio=args.mask_ratio)

            with torch.no_grad():
                t_image_features, t_text_features, t_logit_scale = t_model(images, texts)

            losses = loss(image_features, text_features, logit_scale, \
                          t_image_features, t_text_features, t_logit_scale)

            task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss = losses
            total_loss = task_loss + ckd_loss + icl_loss + cross_kd_loss + fd_loss + gd_loss + afd_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_task.update(task_loss.item(), batch_size)
            loss_icl.update(icl_loss.item(), batch_size)
            loss_ckd.update(ckd_loss.item(), batch_size)
            loss_cross_kd.update(cross_kd_loss.item(), batch_size)
            loss_fd.update(fd_loss.item(), batch_size)
            loss_gd.update(gd_loss.item(), batch_size)
            loss_afd.update(afd_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Total Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Task Loss: {loss_task.val:#.5g} ({loss_task.avg:#.4g}) "
                f"ICL Loss: {loss_icl.val:#.5g} ({loss_icl.avg:#.4g}) "
                f"CKD Loss: {loss_ckd.val:#.5g} ({loss_ckd.avg:#.4g}) "
                f"Cross KD Loss: {loss_cross_kd.val:#.5g} ({loss_cross_kd.avg:#.4g}) "
                f"FD Loss: {loss_fd.val:#.5g} ({loss_fd.avg:#.4g}) "
                f"GD Loss: {loss_gd.val:#.5g} ({loss_gd.avg:#.4g}) "
                f"AFD Loss: {loss_afd.val:#.5g} ({loss_afd.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size * args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def evaluate2(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    # Conserve l'évaluation zero-shot originale si elle est configurée
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []

        # Extraction du biais du modèle s'il existe pour l'utiliser dans la loss
        model_unwrapped = unwrap_model(model)
        has_bias = hasattr(model_unwrapped, 'logit_bias')
        bias_val = model_unwrapped.logit_bias.item() if has_bias else 0.0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)

                    # Stockage des features pour le calcul des métriques finales (R@k)
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())

                    # Extraction du logit_scale moyen du batch
                    current_scale = logit_scale.mean().item()

                    # --- CALCUL DE LA LOSS SIGLIP (PAIRE PAR PAIRE) ---
                    # 1. Calcul de la matrice de similarité brute
                    similarity = image_features @ text_features.t()

                    # 2. Application du scale et du biais apprenable
                    logits_per_image = (current_scale * similarity) + bias_val

                    # 3. Création de la matrice de labels SigLIP (+1 pour les vrais couples, -1 pour les négatifs)
                    batch_size = images.shape[0]
                    labels = torch.eye(batch_size, device=device) * 2.0 - 1.0

                    # 4. Perte logsigmoïde binaire stable
                    # Formule : -log(sigmoid(logits * labels))
                    total_loss = -F.logsigmoid(logits_per_image * labels).sum() / batch_size

                cumulative_loss += total_loss.item() * batch_size
                num_samples += batch_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss (SigLIP): {cumulative_loss / num_samples:.6f}\t")

            # Calcul des métriques globales de Retrieval (R@1, R@5, R@10, etc.)
            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=torch.tensor(current_scale),  # Utilise le dernier scale extrait
            )

            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss, "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics
    
def evaluate1(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)
    

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

        # --- VISUALISATIONS PNG ---
    if is_master(args):
        dataset = data['val'].dataloader.dataset
        captions = dataset.captions  # ✓ attribut confirmé dans CsvDataset
        image_paths = [os.path.join(dataset.root, str(p)) for p in
                        dataset.images]  # ✓ dataset.root est le data_root

        img_feats = torch.cat(all_image_features)
        txt_feats = torch.cat(all_text_features)

        save_similarity_heatmap(img_feats, txt_feats, logit_scale.cpu(), captions, epoch, args)
        save_retrieval_grid(image_paths, captions, img_feats, txt_feats, logit_scale.cpu(), epoch, args)

    return metrics


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                                         F.cross_entropy(logits_per_image, labels) +
                                         F.cross_entropy(logits_per_text, labels)
                                 ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics

def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def save_similarity_heatmap(image_features, text_features, logit_scale, captions, epoch, args):
    """Heatmap matrice de similarité image x texte."""
    logits = (logit_scale * image_features @ text_features.t()).detach().cpu().numpy()
    n = len(captions)

    # Tronquer les captions pour l'affichage
    short_captions = [c[:40] + "..." if len(c) > 40 else c for c in captions]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), max(8, n * 0.4)))
    im = ax.imshow(logits, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Similarité')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_captions, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([f"img_{i}" for i in range(n)], fontsize=7)
    ax.set_xlabel("Textes")
    ax.set_ylabel("Images")
    ax.set_title(f"Matrice de similarité image×texte — Époque {epoch}")

    # Mettre en évidence la diagonale (ground truth)
    for i in range(n):
        ax.add_patch(patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                     linewidth=2, edgecolor='red', facecolor='none'))

    plt.tight_layout()
    out_dir = os.path.join(args.checkpoint_path, "viz")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"heatmap_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_retrieval_grid(image_paths, captions, image_features, text_features,
                        logit_scale, epoch, args, top_k=3, max_images=10):
    """Grille : pour chaque image, affiche le top-k textes prédits."""
    logits = (logit_scale * image_features @ text_features.t()).detach().cpu()
    n = min(len(image_paths), max_images)

    fig, axes = plt.subplots(n, 1, figsize=(14, n * 2.2))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        # Charger l'image
        try:
            img = PILImage.open(image_paths[i]).convert("RGB")
            img.thumbnail((120, 120))
        except Exception:
            img = PILImage.fromarray(np.zeros((120, 120, 3), dtype=np.uint8))

        # Top-k prédictions
        scores, indices = torch.topk(logits[i], k=min(top_k, len(captions)))
        gt_idx = i  # ground truth = diagonale

        ax.axis('off')
        # Afficher l'image à gauche
        ax_img = fig.add_axes([0.01, 1 - (i + 1) / n + 0.01, 0.12, 0.9 / n])
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f"img_{i}", fontsize=7)

        # Texte des prédictions
        lines = []
        for rank, (score, idx) in enumerate(zip(scores.tolist(), indices.tolist())):
            marker = "✓" if idx == gt_idx else "✗"
            caption = captions[idx][:80] + "..." if len(captions[idx]) > 80 else captions[idx]
            lines.append(f"{marker} #{rank+1} (score={score:.2f}) — {caption}")

        text_block = "\n".join(lines)
        color = "green" if indices[0].item() == gt_idx else "red"
        ax.text(0.15, 0.5, text_block, transform=ax.transAxes,
                fontsize=8, va='center', ha='left',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))
        ax.set_title(f"Image {i} — top-1 {'✓ correct' if indices[0].item() == gt_idx else '✗ incorrect'}",
                     fontsize=8, color=color)

    fig.suptitle(f"Retrieval image→texte — Époque {epoch}", fontsize=11, y=1.01)
    out_dir = os.path.join(args.checkpoint_path, "viz")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"retrieval_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
