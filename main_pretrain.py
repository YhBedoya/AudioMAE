import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchaudio.transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

from util import DatasetGenerator

assert timm.__version__ == "0.3.2"  # version check

"""
The following is a necessary fix in the file /usr/local/lib/python3.7/dist-packages/timm/models/layers/helpers.py

import torch
from itertools import repeat
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
"""

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_audio_mae

from engine_pretrain import train_one_epoch

import torch
from torch import nn
from functools import partial
from models_audio_mae import AudioMaskedAutoencoderViT

#audio_mels = torch.ones([2, 1, 1024, 128]) expected shape

def get_args_parser():
    parser = argparse.ArgumentParser('audioMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int) #TODO: check in the original the number of epochs
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='audioMae_vit_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=64, type=int, #TODO: define according to the paper parameters
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float, #TODO: define according to the paper parameters
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, #TODO: define according to the paper parameters
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR', #TODO: define according to the paper parameters
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', #TODO: define according to the paper parameters
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', #TODO: define according to the paper parameters
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', #TODO: define according to the paper parameters
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/content/drive/MyDrive/Data Science and Engineering - PoliTo2/Thesis/models/mae-main/Data/', type=str,  #TODO: data dir
                        help='dataset path')

    parser.add_argument('--output_dir', default='/content/drive/MyDrive/Data Science and Engineering - PoliTo2/Thesis/models/mae-main/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/content/drive/MyDrive/Data Science and Engineering - PoliTo2/Thesis/models/mae-main/output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',  #TODO:cuda
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int) #TODO: 10
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    #TODO: Define the data augmentation according to the paper
    """ 
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    """

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate= 16000, #Define,
        n_fft = 1024,
        hop_length = 512,
        n_mels=64
    )
    sample_rate = 16000  #TODO: Define this in the parameters
    dataset_train = DatasetGenerator(args.data_path, mel_spectrogram, sample_rate)

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train')) #TODO: datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)