import os
import ast
import argparse
from torch.utils.tensorboard import SummaryWriter
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from monai.config import print_config
from monai.metrics import DiceMetric
import numpy as np

from css.utils import generate_data_list
from monai.transforms import AsDiscrete
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
import torch
from monai.utils import set_determinism
from css.bhsd_dataset import generate_data
from css.bhsd_model import css_model
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def setup_distributed_training():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    print(f"local rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed_training():
    dist.destroy_process_group()

def train(train_loader, val_loader, args, writer, model):
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.eta_min)
    scaler = torch.cuda.amp.GradScaler()
    dice_val_best = 0.0
    cls_dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    model.train()
    epoch_loss = 0
    step = 0

    for epoch_idx in range(args.epoch):
        train_loader.sampler.set_epoch(epoch_idx)  # Set epoch for DistributedSampler
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True) if args.local_rank == 0 else train_loader
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].to(args.device), batch["label"].to(args.device))
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if args.local_rank == 0:
                epoch_iterator.set_description(  
                    f"Training ({epoch_idx} / {args.epoch} Steps) (loss={loss:2.5f})"
                )
            if (step % args.eval_num == 0 and step != 0) or epoch_idx == args.epoch:
                if args.local_rank == 0:  # Only validate on the main process
                    dice_val, mean_dice_val_without_bg, class_dice_vals = validation(model, val_loader, epoch_idx, args)
                    writer.add_scalar('Validate/mean_Dice_with_bg', dice_val, epoch_idx)
                    writer.add_scalar('Validate/mean_Dice_without_bg', mean_dice_val_without_bg, epoch_idx)
                    for i, dice_score in enumerate(class_dice_vals):
                        writer.add_scalar(f'Validate/{args.name_list[i]}_Dice', dice_score, epoch_idx)
                    epoch_loss /= step
                    epoch_loss_values.append(epoch_loss)
                    metric_values.append(mean_dice_val_without_bg)
                    if mean_dice_val_without_bg > dice_val_best:
                        dice_val_best = mean_dice_val_without_bg
                        cls_dice_val_best = class_dice_vals
                        global_step_best = epoch_idx
                        torch.save(model.state_dict(), os.path.join(args.exp_dir, f'epoch_{epoch_idx}_dice:{mean_dice_val_without_bg:.5f}.pth'))
                        print(
                            "Model Was Saved ! Current Best Avg. Dice: {} Current Best cls. Dice: {} Current Avg. Dice: {} cls. Dice: {}".format(dice_val_best, cls_dice_val_best, mean_dice_val_without_bg, class_dice_vals))
                    else:
                        print(
                            "Model Was Not Saved ! Current Best Avg. Dice: {} Current Best cls. Dice: {} Current Avg. Dice: {} cls. Dice: {}".format(dice_val_best, cls_dice_val_best, mean_dice_val_without_bg, class_dice_vals))
        scheduler.step()
        if args.local_rank == 0:  # Only log on the main process
            writer.add_scalar('Train/Loss', loss.item(), epoch_idx)
            writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch_idx)

    return dice_val_best, global_step_best, epoch_loss_values, metric_values


# ******************************************train*************************************************************

def validation(model, val_loader, epoch_idx, args):
    model.eval()
    post_label = AsDiscrete(to_onehot=6)
    post_pred = AsDiscrete(argmax=True, to_onehot=6)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, num_classes=6)
    classwise_dice = DiceMetric(include_background=True, reduction='none', get_not_nans=False, num_classes=6)
    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True) if args.local_rank == 0 else val_loader

    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].to(args.device), batch["label"].to(args.device))
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, args.ref_window, 2, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            classwise_dice(y_pred=val_output_convert, y=val_labels_convert)
            if args.local_rank == 0:
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (epoch_idx, len(epoch_iterator_val))) 
        mean_dice_val_include_bg = dice_metric.aggregate().item()
        classwise_dice_val = classwise_dice.aggregate().cpu()
        dice_values = torch.sum(torch.nan_to_num(classwise_dice_val, nan=0.0), dim=0) / torch.tensor([96, 13, 62, 54, 51, 35])
        mean_dice_val_without_bg = dice_values[1:].mean().item()
        dice_metric.reset()
        classwise_dice.reset()

    return mean_dice_val_include_bg, mean_dice_val_without_bg, dice_values.numpy()



# ******************************************Plot the loss and metric******************************************
def draw_fig(epoch_loss_values, metric_values, args):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [args.eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [args.eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.savefig(args.fig_save_path)
    plt.show()
# ******************************************Plot the loss and metric******************************************


def count_class_num(val_loader):
    count_dict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}  
    for batch_data in val_loader:
        label = batch_data['label'].cpu().numpy()
        count_list = np.unique(label)
        for i in count_dict.keys():
            if int(i) in count_list:
                count_dict[i] += 1
    return count_dict
        
        
        
def main():
    paser = argparse.ArgumentParser(description="Train Swin_unetr with BHSD dataset")
    paser.add_argument('--exp_dir', default='css/experiment/swim_unetr/11.28', help="dir of saving files")
    paser.add_argument('--data_path', default='BSHD_src_data/preprocessed_image')
    paser.add_argument('--epoch', default=500)
    paser.add_argument('--eval_num', type=int, default=96)
    paser.add_argument('--model', type=str, choices=['swin_unetr', 'swin_unetr_css_merging'], default="swin_unetr")
    paser.add_argument('--seed', default=42)
    paser.add_argument('--fig_save_name', default='css/train.png', help="name of saving fig")
    paser.add_argument('--lr', default=1e-4, type=float, help="start learning rate")
    paser.add_argument('--batch_size', type=int, default=2)
    paser.add_argument('--grad_accumulate_step', type=int, default=4)
    paser.add_argument('--weight_decay', default=1e-4)
    paser.add_argument('--num_workers', default=0)  # 大于0会与os.environ['CUDA_LAUNCH_BLOCKING'] = "1"冲突
    paser.add_argument('--test', action='store_true')
    paser.add_argument('--eta_min', type=float, default=1e-5, help='min learning rate')
    paser.add_argument('--transforms', choices=['my_tr_trs', 'source_tr_trs', 'css_tr_trs'], default='css_tr_trs')
    paser.add_argument('--spacing', default=(1.5, 1.5, 2.0), type=ast.literal_eval, help="spacing for transforms, Use the form ''(a, b, c)'' to specify")
    paser.add_argument('--ref_window', default=(96, 96, 32), type=ast.literal_eval, help='reference window size & data_processing size for RandCropByPosNegLabeld')
    paser.add_argument('--merging_type', choices=['maxpool', 'avgpool', 'maxavgpool', 'conv'], default=None)
    paser.add_argument('--use_ln', action='store_true', help='if specify, use LayerNorm for conv-Merging, else use InstanceNorm')
    paser.add_argument('--ref_weight', default=None, help='path of trained model')
    paser.add_argument('--use_1x1_conv_for_skip', action='store_true', help='use 1x1 conv3d to change channel in skip connection')
    paser.add_argument('--css_skip', action='store_true', help='using css skip connection')
    paser.add_argument('--use_css_skip_m4', action='store_true', help='using css skip connection m4')
    paser.add_argument('--use_css_skip_m1V2', action='store_true', help='using css skip connection m1v2')
    paser.add_argument('--device', type=str, default="1,2", help='using gpu device for train')

    args = paser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    args.local_rank = setup_distributed_training()
    device = torch.device(f"cuda:{args.local_rank}")
    args.device = device

    num_device = torch.cuda.device_count()
    args.num_device = num_device

    os.makedirs(args.exp_dir, exist_ok=True)
    log_dir = os.path.join(args.exp_dir, "logs")
    args.fig_save_path = os.path.join(args.exp_dir, args.fig_save_name)
    writer = SummaryWriter(log_dir=log_dir) if args.local_rank == 0 else None

    train_ds, train_loader, val_ds, val_loader = generate_data(args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)

    train_loader = torch.utils.data.DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    model = css_model(args).to(device)
    model = DDP(model, device_ids=[args.local_rank])

    set_track_meta(False)
    set_determinism(seed=args.seed)

    if args.test:
        if args.local_rank == 0:
            print("Running in test mode...")
        mean_dice_val_include_bg, mean_dice_val_without_bg, dice_values = validation(model, val_loader, 0, args)
        if args.local_rank == 0:
            print(f"Validation completed. Mean Dice (with background): {mean_dice_val_include_bg:.4f}, Mean Dice (without background): {mean_dice_val_without_bg:.4f}")
            for i, dice_val in enumerate(dice_values):
                print(f"Class {i} Dice: {dice_val:.4f}")
    else:
        dice_val_best, global_step_best, epoch_loss_values, metric_values = train(train_loader, val_loader, args, writer=writer, model=model)

        if args.local_rank == 0:
            print(f"Train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")
            draw_fig(epoch_loss_values, metric_values, args)
    cleanup_distributed_training()

if __name__ == '__main__':
        main()
