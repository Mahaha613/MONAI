import os
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def train(train_loader, val_loader, args):
    torch.backends.cudnn.benchmark = True
    model = css_model(args.model, device=args.device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for epoch_idx in range(args.epoch):
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            if 6 in torch.unique(y):
                print(f'label: {torch.unique(y)}')
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_iterator.set_description(  # noqa: B038
                f"Training ({epoch_idx} / {args.epoch} Steps) (loss={loss:2.5f})"
            )
            if (epoch_idx % args.eval_num == 0 and epoch_idx != 0) or epoch_idx == args.epoch:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(model, epoch_iterator_val, epoch_idx)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = epoch_idx
                    torch.save(model.state_dict(), os.path.join(args.root_dir, "best_metric_model.pth"))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
    return dice_val_best, global_step_best, epoch_loss_values, metric_values
# ******************************************train*************************************************************

def validation(model, epoch_iterator_val, epoch_idx):
    model.eval()
    post_label = AsDiscrete(to_onehot=6)
    post_pred = AsDiscrete(argmax=True, to_onehot=6)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 32), 2, model)  # torch.Size([1, 5, 170, 170, 78])
            val_labels_list = decollate_batch(val_labels)  # torch.Size([1, 170, 170, 78]) 去除batch维度
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (epoch_idx, 10.0))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


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

def main():
    paser = argparse.ArgumentParser(description="Train Swin_unetr with BHSD dataset")
    # paser.add_argument('--device', default=0, required=True, help="choose a device for train, witch can be an int or list or 'cpu'")
    paser.add_argument('--root_dir', default='css/experiment/swim_unetr/11.14', help="dir of saving files")
    paser.add_argument('--epoch', default=30000)
    paser.add_argument('--eval_num', default=500)
    paser.add_argument('--model', choices=['swin_unetr'], default='swin_unetr')
    paser.add_argument('--seed', default=42)
    paser.add_argument('--fig_save_path', default='css/train.png')
    paser.add_argument('--lr', default=1e-4)
    paser.add_argument('--weight_decay', default=1e-5)
    args = paser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    os.makedirs(args.root_dir, exist_ok=True)
    set_determinism(seed=args.seed)
    train_ds, train_loader, val_ds, val_loader = generate_data()
    set_track_meta(False)  
    dice_val_best, global_step_best, epoch_loss_values, metric_values = train(train_loader, val_loader, args)
    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")

    draw_fig(epoch_loss_values, metric_values, args)

if __name__ == '__main__':
    main()
    