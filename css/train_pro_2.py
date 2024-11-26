import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
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


def train(train_loader, val_loader, args, writer):
    torch.backends.cudnn.benchmark = True
    model = css_model(args.model, args, device=args.device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    model.train()
    epoch_loss = 0
    step = 0
    # epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for epoch_idx in range(args.epoch):
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
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
            writer.add_scalar('Train/Loss', loss.item(), epoch_idx * len(train_loader) + step)
            epoch_iterator.set_description(  
                f"Training ({epoch_idx} / {args.epoch} Steps) (loss={loss:2.5f})"
            )
            if (step % args.eval_num == 0 and step != 0) or epoch_idx == args.epoch:
                # epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val, mean_dice_val_without_bg, class_dice_vals = validation(model, val_loader, epoch_idx, args)
                writer.add_scalar('Validate/mean_Dice_with_bg', dice_val, epoch_idx)
                writer.add_scalar('Validate/mean_Dice_without_bg', mean_dice_val_without_bg, epoch_idx)
                for i, dice_score in enumerate(class_dice_vals):
                    writer.add_scalar(f'Validate/{args.name_list[i]}_Dice', dice_score, epoch_idx)

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = epoch_idx
                    torch.save(model.state_dict(), os.path.join(args.root_dir, f'{epoch_idx}_best_metric_model.pth'))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
        scheduler.step()
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch_idx)


    return dice_val_best, global_step_best, epoch_loss_values, metric_values
# ******************************************train*************************************************************


def validation(model, val_loader, epoch_idx, args):
    model.eval()
    post_label = AsDiscrete(to_onehot=6)
    post_pred = AsDiscrete(argmax=True, to_onehot=6)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, num_classes=6)
    classwise_dice = DiceMetric(include_background=True, reduction='none', get_not_nans=False, num_classes=6)
    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)

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
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (epoch_idx, len(epoch_iterator_val))) 
        mean_dice_val_include_bg = dice_metric.aggregate().item()
        classwise_dice_val = classwise_dice.aggregate().cpu()
        # dice_values = torch.sum(torch.nan_to_num(classwise_dice_val, nan=0.0), dim=0) / ((classwise_dice_val >= 0).sum(dim=0))
        dice_values = torch.sum(torch.nan_to_num(classwise_dice_val, nan=0.0), dim=0) / torch.tensor([96, 13, 62, 54, 51, 35])
        # {'1': 13, '2': 62, '3': 54, '4': 51, '5': 35}
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
    # paser.add_argument('--device', default=0, required=True, help="choose a device for train, witch can be an int or list or 'cpu'")
    paser.add_argument('--root_dir', default='css/experiment/swim_unetr/11.262', help="dir of saving files")
    paser.add_argument('--data_path', default='BSHD_src_data/preprocessed_image')
    paser.add_argument('--epoch', default=300)
    paser.add_argument('--eval_num', default=96)
    paser.add_argument('--model', choices=['swin_unetr'], default='swin_unetr')
    paser.add_argument('--seed', default=42)
    paser.add_argument('--fig_save_path', default='css/train_pro.png')
    paser.add_argument('--lr', default=1e-3, help="start learning rate")
    paser.add_argument('--batch_size', default=1)
    paser.add_argument('--weight_decay', default=1e-4)
    paser.add_argument('--num_workers', default=0)  # 大于0会与os.environ['CUDA_LAUNCH_BLOCKING'] = "1"冲突
    paser.add_argument('--test', action='store_true')
    paser.add_argument('--transforms', choices=['my_tr_trs', 'source_tr_trs'], default='source_tr_trs')
    paser.add_argument('--ref_window', default=(96, 96, 32))
    paser.add_argument('--merging_type', choices=['maxpool', 'avgpool', 'maxavgpool', 'conv'], default=None)
    # paser.add_argument('--scehduler', action='store_true')
    args = paser.parse_args()
    
    log_dir = os.path.join(args.root_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")
    args.device = device
    name_list = ['BG', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH']
    args.name_list = name_list
    os.makedirs(args.root_dir, exist_ok=True)
    set_determinism(seed=args.seed)
    train_ds, train_loader, val_ds, val_loader = generate_data(args)

    # class_list = count_class_num(val_loader)
    # print(class_list)  # {'1': 13, '2': 62, '3': 54, '4': 51, '5': 35}
    
    set_track_meta(False)
    if not args.test:
        dice_val_best, global_step_best, epoch_loss_values, metric_values = train(train_loader, val_loader, args, writer=writer)
        print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")
        draw_fig(epoch_loss_values, metric_values, args)
    else:
        weight_path = glob(os.path.join(args.root_dir, "best*.pth"))
        assert os.path.isfile(weight_path), "weight path is not a file"
        model = css_model(args.model, device)
        model.load_state_dict(torch.load(weight_path))
        mean_dice_val_include_bg, mean_dice_val_without_bg, dice_values = validation(model, val_loader, 0)
        print(f"mean_dice_val_include_bg: {mean_dice_val_include_bg:.4f}, mean_dice_val_without_bg: {mean_dice_val_without_bg:.4f}, dice_values: {dice_values}")


if __name__ == '__main__':
        main()
