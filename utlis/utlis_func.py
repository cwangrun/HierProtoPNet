import os
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data.xray_dataloader import ChestDataset
from data.augmentations.rand_augment import RandAugment

train_h, train_w = 512, 512


def get_aug_trans(n=2, m=9, min_scale=0.8):  # s = 0.08
    return transforms.Compose([
        RandAugment(n=n, m=m),
        transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.65, 1.0), ratio=(0.65, 1.35)),
        transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.9, 1.1), (-0.01, 0.01)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-25, 25, -25, 25), fill=0),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def config_dataset(params):
    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(params.seed)  # python random seed
    np.random.seed(params.seed)  # numpy random seed
    torch.manual_seed(params.seed)  # pytorch random seed
    torch.cuda.manual_seed(params.seed)  # pytorch random seed
    torch.backends.cudnn.deterministic = True

    transform_train = get_aug_trans(n=1, m=30)  # m=9

    transform_push = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test_val = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = params.root_dir

    train_dataset = ChestDataset(root_dir=root_dir, transform=transform_train, mode='train')
    train_push_dataset = ChestDataset(root_dir=root_dir, transform=transform_push, mode='train')

    test_dataset = ChestDataset(root_dir=root_dir, transform=transform_test_val, mode='test')
    valid_dataset = ChestDataset(root_dir=root_dir, transform=transform_test_val, mode='test')

    def make_weights_for_balanced_classes(images, nclasses):
        count = images.sum(axis=0).tolist()
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])

        # [12.617149758454106,
        #  61.200937316930286,
        #  12.064903568541402,
        #  7.580177042519228,
        #  25.897372335151214,
        #  22.189889549702634,
        #  119.2579908675799,
        #  39.616989002654535,
        #  36.630434782608695,
        #  75.81277213352685,
        #  73.41531974701336,
        #  83.50919264588329,
        #  46.59678858162355,
        #  740.9219858156029,
        #  2.0687128712871288]

        weight_per_class = [12.617149758454106,
                            61.200937316930286,
                            12.064903568541402,
                            7.580177042519228,
                            25.897372335151214,
                            22.189889549702634,
                            100,
                            39.616989002654535,
                            36.630434782608695,
                            75.81277213352685,
                            73.41531974701336,
                            83.50919264588329,
                            46.59678858162355,
                            150,  # 150
                            2.0687128712871288 * 2  # 2
                            ]

        weight_per_class = np.array(weight_per_class)
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = np.mean(weight_per_class[val == 1]).item()
        return weight

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(train_dataset.gr, params.num_classes)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, shuffle=False, sampler=sampler, drop_last=False, num_workers=0)    # shuffle=True # 8
    train_push_loader = DataLoader(train_push_dataset, batch_size=params.train_push_batch_size, shuffle=True, drop_last=False, num_workers=16)    # 8
    test_loader = DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=0)     # 16
    valid_loader = DataLoader(valid_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=16)

    print('Number of train images: {}'.format(len(train_dataset)))
    print('Number of train push images: {}'.format(len(train_push_dataset)))
    print('Number of test images: {}'.format(len(test_dataset)))
    print('Number of valid images: {}'.format(len(valid_dataset)))
    print('')

    return train_loader, train_push_loader, test_loader, valid_loader

