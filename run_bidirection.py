import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from util_tools.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from util_tools.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from dataset.datasets import build_dataset
from util_tools.utils import NativeScalerWithGradNormCount as NativeScaler, load_bidir_weights, unfreeze_block
from util_tools.utils import cross_multiple_samples_collate, laod_eval_weights
import util_tools.utils as utils
import models.bidir_modeling_crossattn

from loss_functions.triplet import TripletLoss
from loss_functions.triplet_loss import BatchAllTtripletLoss
from custom_sampler import CustomBatchSampler




def get_args():
    parser = argparse.ArgumentParser('CAST fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1,type=int)
    parser.add_argument('--vote_strategy', default='maxlogit', type=str)

    # Model parameters
    parser.add_argument('--vmae_model', default='bidir_vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--clip_frame', default=None, type=str)
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.3, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--head_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate for head (default: 0.)')
    
    parser.add_argument('--down_ratio', type=int, default=2, metavar='PCT',
                        help='B-CAST module down projection ratio')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=(0.9,0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-3, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=2)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup e		nabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=0.9,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--vmae_finetune', default='../sthv2/sthv2/checkpoint.pth', help='finetune from vmae checkpoint')
    parser.add_argument('--clip_finetune',default='../sthv2/sthv2/ViT-B-16.pt', help='finetune from clip checkpoint')
    #parser.add_argument('--fine_tune',default="../sthv2/sthv2/CAST_SSV2_50epoch_71.6_noextra.pt", help='finetune from bidir model')
    parser.add_argument('--fine_tune',default='../sthv2/sthv2/output_selector/checkpoint-7.pth', help='finetune from bidir model')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=1.0, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='../sthv2/sthv2/videos', type=str,
                        help='dataset path')
    parser.add_argument('--anno_path', default='../sthv2/sthv2/annotations/labels', type=str, help='annotation path')
    parser.add_argument('--eval_data_path', default='../sthv2/sthv2/videos', type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=174, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='SSV2', choices=['Kinetics-400', 'SSV2','image_folder', 'EPIC', 'UCF101'],
                        type=str, help='dataset')
    parser.add_argument('--pred_type', default=None, choices=['noun', 'verb', 'action'])
    parser.add_argument('--output_dir', default='../sthv2/sthv2/output_selector',
                        help='path where to tensorboard log')
    parser.add_argument('--log_dir', default='../sthv2/sthv2/logs',
                        help='path where to tensorboard log')
                        
    parser.add_argument('--device', default='cuda:5',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=5, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0,type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    # new settings
    parser.add_argument('--freeze_layers', default=None, nargs='+', type=str)
    parser.add_argument('--slack_api', type=str,default=None)
    parser.add_argument('--composition', action='store_true')
    parser.add_argument('--class_mapping', default='class_grouping.json', type=str,
                        help='Path to the class to head mapping JSON file')
    

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init
    
def load_selector(selector_ckpt_path, model_name,load_eval_weights, args):
    selector = create_model(
        model_name,
        pretrained=False,
        num_classes=15,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        down_ratio=args.down_ratio,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )

    
    #from util_tools.utils import load_eval_weights
    load_eval_weights(selector, selector_ckpt_path, args)
    
    selector.to(args.device)
    selector.eval()
    return selector
    

def load_expert_models(model_name, ckpt_dir,load_eval_weights, unfreeze_block, args):
    experts = []
    

    for i in range(15):
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            down_ratio=args.down_ratio,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            head_drop_rate=args.head_drop_rate,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )
        #print("list of ",os.listdir(ckpt_dir))
        ckpt_path = os.path.join(ckpt_dir, f"expert-{i}.pth")

        
        load_eval_weights(model, ckpt_path, args)
        

        model, _ = unfreeze_block(model, ['cross', 'clip_space_time_pos', 'clip_time_pos', 'vmae_time_pos', 'Adapter', 'ln_post', 'vmae_fc_norm','last_proj', 'fusion', 'head'])
        model.to(args.device)
        model.eval()
        experts.append(model)
    return experts
    
def load_class_to_head_mapping(mapping_file):
    """Load the mapping from class to head labels from a JSON file."""
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Convert string keys to integers
    class_to_head = {int(k): v for k, v in mapping_data['class_to_head'].items()}
    return class_to_head

class HeadLabelWrapper:
    """Wrapper for dataset that converts class labels to head labels."""
    def __init__(self, dataset, class_to_head_map):
        self.dataset = dataset
        self.class_to_head_map = class_to_head_map
        
        # If the dataset has a label_array attribute, convert it
        if hasattr(dataset, 'label_array'):
            self.label_array = np.array([self.class_to_head_map.get(int(label), 0) 
                                        for label in dataset.label_array])
    
    def __getitem__(self, index):
        data = self.dataset[index]
        # Typically dataset returns (video, label) or similar
        # Modify the label part to use head label
        if isinstance(data, tuple) and len(data) >= 2:
            video, label = data[0], data[1]
            # Convert the class label to head label
            head_label = self.class_to_head_map.get(int(label), 0)
            # Return the modified tuple
            return (video, head_label, *data[2:])
        return data
    
    def __len__(self):
        return len(self.dataset)
        
def main(args, ds_init):
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)
    
    print(args)

    device = torch.device(args.device)
    #device='cpu'
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    print("OK1")
    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    print("ok2")
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    
    
    ## uncomment the docstring when training selector
    '''
    print("Loading class to head mapping from", args.class_mapping)
    class_to_head_map = load_class_to_head_mapping(args.class_mapping)
    
    # Wrap the datasets to convert class labels to head labels
    dataset_train = HeadLabelWrapper(dataset_train, class_to_head_map)
    if dataset_val is not None:
        dataset_val = HeadLabelWrapper(dataset_val, class_to_head_map)
    if dataset_test is not None:
        dataset_test = HeadLabelWrapper(dataset_test, class_to_head_map)
    
    args.nb_classes = len(set(class_to_head_map.values()))
    print(f"Updated number of classes (heads): {args.nb_classes}")
    
    '''  
    print("ok3")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    #sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_train = CustomBatchSampler(dataset_train.label_array, batch_size=args.batch_size)
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(cross_multiple_samples_collate, fold=False)
    else:
        collate_func = None
    '''
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )
    '''
    
    
    
    data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_sampler=sampler_train,  # Use batch_sampler instead of sampler
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    collate_fn=collate_func,  # Keep collate_fn if needed
)
    

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes, composition=args.composition)

    patch_size = 14
    print("Patch size = %s" % str(patch_size))
    args.window_size = 16
    args.patch_size = patch_size
    
    
    
    model = create_model(
          args.vmae_model,
          pretrained=False,
          num_classes=args.nb_classes,
          
          all_frames=args.num_frames * args.num_segments,
          tubelet_size=args.tubelet_size,
          down_ratio=args.down_ratio,
          drop_rate=args.drop,
          drop_path_rate=args.drop_path,
          attn_drop_rate=args.attn_drop_rate,
          head_drop_rate=args.head_drop_rate,
          drop_block_rate=None,
          use_mean_pooling=args.use_mean_pooling,
          init_scale=args.init_scale,
      )
    
    if args.fine_tune is not None:
        laod_eval_weights(model, args.fine_tune, args)
        print('fine main checkpoints', args.fine_tune)
    else:
        load_bidir_weights(model, args)
        print('fine bidir checkpoints', args.fine_tune)
    
    model, unfreeze_list = unfreeze_block(model, ['cross', 'clip_space_time_pos', 'clip_time_pos', 'vmae_time_pos', 'Adapter', 'ln_post', 'vmae_fc_norm','last_proj', 'fusion', 'head'])
    #print('unfreeze list :', unfreeze_list)
    
    #print(model, "-------")
    model.to(device)
    
    
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    total_batch_size=args.batch_size
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = len(model_without_ddp.blocks)
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        #print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        #criterion = SoftTargetCrossEntropy()
        ##Changed the loss 
        
        criterion=BatchAllTtripletLoss() 
        criterion2= SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion2))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
        
    
    
    if args.composition:
        from engine_for_compomodel import train_one_epoch, validation_one_epoch, final_test, merge
    else:
        from engine_for_onemodel import train_one_epoch, validation_one_epoch, final_test, merge

    if args.eval:
        selector_ckpt_path='../sthv2/sthv2/output_selector/checkpoint-best.pth'
        ckpt_dir='expert_models'
        selector=load_selector(selector_ckpt_path, args.vmae_model,laod_eval_weights, args)
        experts=load_expert_models(args.vmae_model, ckpt_dir,laod_eval_weights,unfreeze_block, args)
        
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        print(preds_file, "preds_file")
        
        test_stats=final_test(args, data_loader_test, selector, experts, device, preds_file) ## use this while testing the full model
        #test_stats = final_test(args, data_loader_test, model, device, preds_file) ## use this when testing individual expert models and selector
        print("test_stats", test_stats)
        
        #torch.distributed.init_process_group(backend="nccl", rank=1)
        #torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1, 
                         'Final Top-5': final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)
        

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    torch.cuda.empty_cache()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            
        torch.autograd.set_detect_anomaly(True)

        train_stats = train_one_epoch(
            args, model, criterion,criterion2, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq
        )
        print(f"you have trained {epoch} epoch")
        torch.cuda.empty_cache()
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = validation_one_epoch(args, data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(args, data_loader_test, model, device, preds_file)
    #torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
