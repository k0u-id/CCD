import os
import sys
import argparse
from munch import Munch as mch
from os.path import join as ospj
from datetime import datetime

_DATASET = ('voc07', 'voc12', 'coco', 'nuswide')
_SCHEMES = ('LL-R', 'LL-Ct', 'LL-Cp')
_LOOKUP = {
    'feat_dim': {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101' : 2048,
    },
    'num_classes': {
        'voc07': 20,
        'voc12': 20,
        'coco': 80,
        'nuswide': 81,
    },
    'delta_rel': {
        'LL-R': 0.5,
        'LL-Ct': 0.2,
        'LL-Cp': 0.1,
    }
}

def set_dir(runs_dir, exp_name, num_epochs):
    runs_dir = ospj(runs_dir, exp_name)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        os.makedirs(os.path.join(runs_dir, 'CAM'))
        for i in range(num_epochs):
            os.makedirs(os.path.join(runs_dir, 'CAM', str(i)))
    return runs_dir

def set_default_configs(args):
    args.ss_seed = 999
    args.ss_frac_train = 1.0
    args.ss_frac_val = 1.0
    args.use_feats = False
    args.val_frac = 0.2
    args.split_seed = 1200
    args.train_set_variant = 'observed'
    args.val_set_variant = 'clean'
    # args.arch = 'resnet50'
    args.freeze_feature_extractor = False
    args.use_pretrained = True
    args.num_workers = 4
    args.lr_mult = 10
    args.ema_decay = 0.8
    args.bank_size = 50
    args.dataset_rate = 0.5
    args.resize_long = (320, 640)
    # args.train_resize = (640, 640)
    args.resize = (640, 640)
    args.train_flip = True
    args.crop_size = 512
    args.crop_method = 'random'
    args.save_path = './results'

    return args

def set_follow_up_configs(args):
    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    now = datetime.now()
    args.experiment_name = str(now).split(".")[0].replace('-','').replace(" ","_").replace(":","")
    args.save_path = set_dir(args.save_path, args.experiment_name, args.num_epochs)

    args.clean_rate = 1

    return args

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=_DATASET)
                        
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--bsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--offset_size', type=int, default=40)
    parser.add_argument('--coeff', type=float, default=0.8)
    parser.add_argument('--loss_coeff', type=float, default=0.1)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--colorjitter', type=bool, default=True)
    parser.add_argument('--update_label', type=str2bool, default=True)
    parser.add_argument('--train_resize', type=list_of_ints, default=(448, 448))
    parser.add_argument('--patch_resize', type=list_of_ints, default=(640, 640))
    parser.add_argument('--box_offset', type=int, default=10)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--LS', type=str2bool, default=False)
    parser.add_argument('--coeff_bias', type=str2bool, default=False)
    parser.add_argument('--method', type=str, default='cam_based', choices=['use_gt', 'gt', 'random', 'cdul', 'cam_based'])
    parser.add_argument('--arch', type=str, default='resnet101')
    parser.add_argument('--inf_num', type=int, default=1)
    parser.add_argument('--global_temp', type=str2bool, default=True)
    parser.add_argument('--local_temp', type=str2bool, default=True)
    parser.add_argument('--use_consist', type=float, default=1)
    parser.add_argument('--LS_coeff', type=int, default=80)
    parser.add_argument('--bound', type=int, default=4)

    args = parser.parse_args()
    args = set_default_configs(args)
    args = set_follow_up_configs(args)
    args = mch(**vars(args))

    return args


