import os
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
import torch
from monai.utils import set_determinism
import matplotlib.pyplot as plt

# 设置确定性行为和随机种子
set_determinism(seed=42)

# ******************************************generate data*****************************************************
def get_transforms(trans, device, is_train=True):
    my_tr_trs = Compose(       
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),  # (h, w, d)
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
            CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),  # (h, w, d)
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 32),  # (H,W,D)
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

    my_val_trs = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            # Lambdad(keys=["image"], func=lambda x: np.clip(x, -4500, 4500)), # z轴上图像和标签维度不一致：image shape: torch.Size([1, 342, 342, 16]), label shape: torch.Size([1, 167, 167, 85])
            # Clip(keys=["image"], min=-4500, max=4500),  # 使用Clip变换来剪裁像素值
            ClipIntensityPercentilesd(keys=["image"], lower=5, upper=95, sharpness_factor=10),
            ScaleIntensityRanged(keys=["image"], a_min=-4500, a_max=4500, b_min=0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ], 
    )

    source_tr_trs = Compose(       
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            # CropForegroundd(keys=["image", "label"], source_key="image", margin=5),  # 消融，功能与RandCropByPosNegLabeld类似
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 2.0),
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

    source_val_trs = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            # CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ], 
    )

    if is_train:
        if 'source' in trans:
            return source_tr_trs
        else:
            return my_tr_trs
    else:
        if 'source' in trans:
            return source_val_trs
        else:
            return my_val_trs


def generate_data(args):
    datalist = generate_data_list(os.path.join(args.data_path, 'train'),
                                'BSHD_src_data/label/train')
    val_files = generate_data_list(os.path.join(args.data_path, 'test'),
                                'BSHD_src_data/label/test')

    train_ds = CacheDataset(
        data=datalist,
        transform=get_transforms(args.transforms, args.device),
        # cache_num=24,
        cache_rate=1.0,
        num_workers=args.num_workers)

    train_loader = ThreadDataLoader(train_ds, 
                                    num_workers=args.num_workers, 
                                    batch_size=args.batch_size, 
                                    shuffle=True)

    val_ds = CacheDataset(
        data=val_files, 
        transform=get_transforms(args.transforms, args.device, is_train=False), 
        # cache_num=24, 
        cache_rate=1.0, 
        num_workers=args.num_workers)

    val_loader = ThreadDataLoader(val_ds, 
                                num_workers=args.num_workers, 
                                batch_size=args.batch_size,
                                # collate_fn=pad_list_data_collate, # batch_size >1 时，collate_fn自动填充不一样的形状
                                )
    set_track_meta(False)
    return train_ds, train_loader, val_ds, val_loader

def vis_data():
    # ******************************************Check data shape and visualize************************************
    train_ds, train_loader, val_ds, val_loader = generate_data()
    case_num = 1
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    label_name = os.path.split(val_ds[case_num]["label"].meta["filename_or_obj"])[1]
    img_shape = img.shape
    label_shape = label.shape
    print(f"{img_name} image shape: {img_shape}, {label_name} label shape: {label_shape}")
    for slice_idx in range(img_shape[-1]):
        if torch.sum(label[0, :, :, slice_idx]) != 0:
            plt.figure("image", (18, 6))
            plt.subplot(1, 2, 1)
            plt.title("image")
            plt.imshow(img[0, :, :, slice_idx].detach().cpu(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("label")
            plt.imshow(label[0, :, :, slice_idx].detach().cpu())
            plt.show()
            break
            
    # ******************************************Check data shape and visualize************************************


# if __name__ == '__main__':
# #     # vis_data()
#     import SimpleITK as sitk
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     source_tr_trs = Compose(       
#             [
#                 LoadImaged(keys=["image", "label"], ensure_channel_first=True),
#                 # CropForegroundd(keys=["image", "label"], source_key="image", margin=5),  # 消融，功能与RandCropByPosNegLabeld类似
#                 # Orientationd(keys=["image", "label"], axcodes="RAS"),
#                 Spacingd(
#                     keys=["image", "label"],
#                     pixdim=(1.0, 1.0, 1.0),
#                     mode=("bilinear", "nearest"),
#                 ),
#                 # EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
#                 # RandCropByPosNegLabeld(
#                 #     keys=["image", "label"],
#                 #     label_key="label",
#                 #     spatial_size=(96, 96, 32),  # (C,H,W,[D])
#                 #     pos=2,
#                 #     neg=1,
#                 #     num_samples=4,
#                 #     image_key="image",
#                 #     image_threshold=0,
#                 # )
#             ]
#         )
#     data_list = [[{"image": "BSHD_src_data/preprocessed_image/train/BHSD_image_000.nii.gz", 
#                    "label": "BSHD_src_data/label/train/BHSD_label_000.nii.gz"}]]
#     train_ds = CacheDataset(
#         data=data_list,
#         transform=source_tr_trs,
#         # cache_num=24,
#         cache_rate=1.0,
#         num_workers=0)
#     train_loader = ThreadDataLoader(train_ds, 
#                                     num_workers=0, 
#                                     batch_size=1, 
#                                     shuffle=True)
#     for train_data in train_loader:
#         print(train_data['image'].shape)

#     img = sitk.ReadImage('BSHD_src_data/preprocessed_image/train/BHSD_image_000.nii.gz')
#     origin = img.GetOrigin()
#     Spacing = img.GetSpacing()
#     direction = img.GetDirection()
#     # print(origin)
#     print(Spacing)
#     # print(direction)
#     img_array = sitk.GetArrayFromImage(img)  #(d, h, w)
#     print(img_array.shape)
    
#     pass
    

