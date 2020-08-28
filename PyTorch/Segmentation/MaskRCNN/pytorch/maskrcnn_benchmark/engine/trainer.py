# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
import herring.torch as herring
#from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from herring.torch import get_world_size

try:
    from apex import amp
    use_amp = False
except ImportError:
    print('Use APEX for multi-precision via apex.amp')
    use_amp = False

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        #dist.reduce(all_losses, dst=0)
        herring.all_reduce(all_losses)
        #if herring.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
        all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    use_amp,
    cfg,
    per_iter_end_callback_fn=None,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    print("max_iter: ", max_iter)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        #loss_dict_reduced = reduce_loss_dict(loss_dict)
        #losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses, **loss_dict)


        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        if use_amp:
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
        else:
            losses.backward()
        if not cfg.SOLVER.ACCUMULATE_GRAD:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            if (iteration + 1) % cfg.SOLVER.ACCUMULATE_STEPS == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(cfg.SOLVER.ACCUMULATE_STEPS)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "avg iteration time(s): {avg_iter:.2f}",
                        "avg iter/s: {iter_s:.2f}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    avg_iter=meters.time.global_avg,
                    iter_s = 1.0/meters.time.global_avg,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            early_exit = per_iter_end_callback_fn(iteration=iteration)
            if early_exit:
                break


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    sec_per_iteration = total_training_time / max_iter
    samples_per_sec = int(cfg.SOLVER.IMS_PER_BATCH) / sec_per_iteration
    logger.info(
        "Total training time: {} ({:.4f} s / it) throughput: {:.2f} FPS".format(
            total_time_str, sec_per_iteration, samples_per_sec
        )
    )
    logger.info("Final Loss at iteration {}: {}".format(max_iter, str(meters)))
