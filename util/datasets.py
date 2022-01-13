# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from neotl.datasets.caffe_lmdb import CaffeLMDB
def build_dataset(is_train, args, linear=False):
    transform = build_transform(is_train, args, linear=linear)
    root = os.path.join(args.data_path, 'train' if is_train else 'valid')
    if args.data_type == "image_folder":
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.data_type == "lmdb":
        dataset = CaffeLMDB(root, transform=transform, label_type="int")
    print(dataset)
    return dataset


def build_transform(is_train, args, linear=False):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if linear:
            normalize = transforms.Normalize(mean=mean, std=std)
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),

                transforms.ToTensor(),
                normalize
            ])
            return transform
        else:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
