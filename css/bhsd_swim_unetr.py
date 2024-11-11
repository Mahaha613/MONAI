import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
import numpy as np

from css.utils import generate_data_list
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    ClipIntensityPercentilesd,
    Lambdad,
)

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
import os 
import torch
from monai.utils import set_determinism

# 设置确定性行为和随机种子
set_determinism(seed=42)

# print_config()
# directory = os.environ.get("MONAI_DATA_DIRECTORY")
# if directory is not None:
#     os.makedirs(directory, exist_ok=True)
# root_dir = tempfile.mkdtemp() if directory is None else directory
# print(root_dir)

# ******************************************generate data*****************************************************
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transforms = Compose(       
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        # 截断
        # Lambdad(keys=["image"], func=lambda x: np.clip(x, a_min=-4000, a_max=4000)),
        ClipIntensityPercentilesd(keys=["image"], lower=5, upper=95, sharpness_factor=10),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-4000,
            a_max=4000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 32),  # (C,H,W,[D])
            pos=2,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        # Lambdad(keys=["image"], func=lambda x: np.clip(x, -4500, 4500)), # z轴上图像和标签维度不一致：image shape: torch.Size([1, 342, 342, 16]), label shape: torch.Size([1, 167, 167, 85])
        # Clip(keys=["image"], min=-4500, max=4500),  # 使用Clip变换来剪裁像素值
        ClipIntensityPercentilesd(keys=["image"], lower=5, upper=95, sharpness_factor=10),
        ScaleIntensityRanged(keys=["image"], a_min=-4500, a_max=4500, b_min=0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ], 
)

datalist = generate_data_list('BSHD_src_data/image/train',
                              'BSHD_src_data/label/train')
val_files = generate_data_list('BSHD_src_data/image/test',
                              'BSHD_src_data/label/test')

train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    # cache_num=24,
    cache_rate=1.0,
    num_workers=8)

train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=2, shuffle=True)
# print(len(train_loader))

val_ds = CacheDataset(
    data=val_files, 
    transform=val_transforms, 
    # cache_num=24, 
    cache_rate=1.0, 
    num_workers=8)

val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

set_track_meta(False)
# ******************************************generate data*****************************************************

# ******************************************Check data shape and visualize************************************
# case_num = 1
# img = val_ds[case_num]["image"]
# label = val_ds[case_num]["label"]
# img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
# label_name = os.path.split(val_ds[case_num]["label"].meta["filename_or_obj"])[1]
# img_shape = img.shape
# label_shape = label.shape
# print(f"{img_name} image shape: {img_shape}, {label_name} label shape: {label_shape}")
# for slice_idx in range(img_shape[-1]):
#     if torch.sum(label[0, :, :, slice_idx]) != 0:
#         plt.figure("image", (18, 6))
#         plt.subplot(1, 2, 1)
#         plt.title("image")
#         plt.imshow(img[0, :, :, slice_idx].detach().cpu(), cmap="gray")
#         plt.subplot(1, 2, 2)
#         plt.title("label")
#         plt.imshow(label[0, :, :, slice_idx].detach().cpu())
#         plt.show()
#         break
        

# ******************************************Check data shape and visualize************************************

# ******************************************Create Swin UNETR model*******************************************
model = SwinUNETR(
    img_size=(64, 64, 64),
    in_channels=1,
    out_channels=6,
    feature_size=48,
    use_checkpoint=True,
    # use_v2=True,
).to(device)

weight = torch.load("css/model_swinvit.pt")
model.load_from(weights=weight)
print("Using pretrained self-supervied Swin UNETR backbone weights !")
# ******************************************Create Swin UNETR model*******************************************

# ******************************************Optimizer and loss function***************************************
torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()

# ******************************************Optimizer and loss function***************************************

# ******************************************train*************************************************************

def validation(epoch_iterator_val):
    model.eval()
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
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
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
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
        )
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best
# ******************************************train*************************************************************





root_dir = 'css/experiment/swim_unetr'
max_iterations = 30000
eval_num = 500
post_label = AsDiscrete(to_onehot=6)
post_pred = AsDiscrete(argmax=True, to_onehot=6)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")

# ******************************************Plot the loss and metric******************************************
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()
# ******************************************Plot the loss and metric******************************************
