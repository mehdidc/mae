# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, step, args):
    sched = args.schedule
    """Decay the learning rate with half-cycle cosine after warmup"""
    if sched == 'cosine':
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs 
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    elif sched == 'rsqrt':
        if step < args.warmup_steps:
            lr = args.lr * step / args.warmup_steps
        else:
            decay_factor = args.lr * math.sqrt(args.warmup_steps)
            lr = decay_factor / math.sqrt(step)
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
