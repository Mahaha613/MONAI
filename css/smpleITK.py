import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from easydict import EasyDict
import numpy as np
from glob import glob
import os
def clip_and_normalize(volume, min_val=-40, max_val=120):
    # 将图像裁剪到指定范围内
    volume = np.clip(volume, min_val, max_val)
    # 将图像标准化到 [0, 1] 范围内
    volume = (volume - 40) / 80
    return volume

def get_data(img_path, save_path):
    img_list = glob(f'{img_path}/*.nii.gz')
    for img in img_list:
        name = os.path.basename(img)
        img = sitk.ReadImage(img)
        src_spacing = img.GetSpacing()
        src_direction = img.GetDirection()
        src_origin = img.GetOrigin()
        data = sitk.GetArrayFromImage(img)
        clip_data = clip_and_normalize(data)
        clip_img = sitk.GetImageFromArray(clip_data)
        clip_img.SetDirection(src_direction)
        clip_img.SetOrigin(src_origin)
        clip_img.SetSpacing(src_spacing)
        sitk.WriteImage(clip_img, f'{save_path}/{name}')

import json
from sklearn.model_selection import KFold

def split_fold_5(path):

    # 假设data是你的数据集，格式为列表
    data = glob(f'{path}/*.nii.gz')

    # 初始化五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 划分数据集
    split_data = {'training':[], 'val':[]}
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(data)):
        for idx in train_indices.tolist():
            name = os.path.basename(data[idx])
            num = os.path.splitext(os.path.splitext(name)[0])[0]
            num = num.split('_')[-1]
            fold_data = {
            'fold': fold_idx,
            'image': [data[idx]],
            'label': f'/home/amax/css/data/BHSD/data_for_Swin_UNETR/label/train/BHSD_label_{num}.nii.gz'
        }
            split_data['training'].append(fold_data)
        for idx in val_indices.tolist():
            name = os.path.basename(data[idx])
            num = os.path.splitext(os.path.splitext(name)[0])[0]
            num = num.split('_')[-1]
            fold_data = {
            'fold': fold_idx,
            'image': [data[idx]],
            'label': f'/home/amax/css/data/BHSD/data_for_Swin_UNETR/label/train/BHSD_label_{num}.nii.gz'
        }
            split_data['val'].append(fold_data)

    # 将划分结果保存为JSON文件
    output_file = '/home/amax/css/data/BHSD/data_for_Swin_UNETR/split.json'
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=4)
        # json.dump(val, f, indent=4)

    print(f"划分结果已保存至 {output_file}")


def mysplit(path):
    data = glob(f'{path}/*.nii.gz')
    idxs = np.arange(96).tolist()
    np.random.shuffle(idxs)
    split_data = {'training':[]}
    for i, idx in enumerate(idxs):
        name = os.path.basename(data[idx])
        num = os.path.splitext(os.path.splitext(name)[0])[0]
        num = num.split('_')[-1]
        fold_data = {
        'fold': i // 19,
        'image': [data[idx]],
        'label': f'/home/amax/css/data/BHSD/data_for_Swin_UNETR/label/train/BHSD_label_{num}.nii.gz'
    }
        split_data['training'].append(fold_data)
    output_file = '/home/amax/css/data/BHSD/data_for_Swin_UNETR/BHSD_split.json'
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=4)


def test_split(path='/home/amax/css/data/BHSD/data_for_Swin_UNETR/image/test'):
    data = glob(f'{path}/*.nii.gz')
    split_data = {'test':[]}
    for test_data in data:
        nfi_img = nib.load(test_data)
        affine = nfi_img.affine
        name = os.path.basename(test_data)
        num = os.path.splitext(os.path.splitext(name)[0])[0]
        num = num.split('_')[-1]
        test_data_ = {
            'image': test_data ,
            'label': f'/home/amax/css/data/BHSD/data_for_Swin_UNETR/label/test/BHSD_label_{num}.nii.gz',
            'label_meta_dict':{"affine":affine.tolist()},
            'image_meta_dict':{"filename_or_obj":name}
        }
        split_data['test'].append(test_data_)
    output_file = '/home/amax/css/data/BHSD/data_for_Swin_UNETR/BHSD_test_split.json'
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=4)


if __name__ == '__main__':
    # get_data(img_path='/home/amax/css/data/BHSD/3d_data_spacing_1_1_1/labels/train',
    #          save_path='/home/amax/css/data/BHSD/data_for_Swin_UNETR/label/train')
    # print(len(os.listdir('/home/amax/css/data/BHSD/data_for_Swin_UNETR/image/train')),
    #       len(os.listdir('/home/amax/css/data/BHSD/data_for_Swin_UNETR/image/test')),
    #       len(os.listdir('/home/amax/css/data/BHSD/data_for_Swin_UNETR/label/test')),
    #       len(os.listdir('/home/amax/css/data/BHSD/data_for_Swin_UNETR/label/train')))
    # with open('/home/amax/css/Swin_UNETR/brats21_folds.json', 'r') as f:
    #     data = json.load(f)
    #     print('.')
    # split_fold_5(path='/home/amax/css/data/BHSD/data_for_Swin_UNETR/image/train')
    # mysplit('/home/amax/css/data/BHSD/data_for_Swin_UNETR/image/train')

    # path = '/home/amax/css/data/BHSD/data_for_Swin_UNETR/image/train'
    # data_list = glob(f'{path}/*.nii.gz')
    # my_list = []
    # with open('/home/amax/css/data/BHSD/data_for_Swin_UNETR/BHSD_split.json') as f:
    #     data = json.load(f)
    #     key = data.keys()
    #     i = 0
    #     for k in key:
    #         for data in data[k]:
    #             i += 1
    #             find = data['image'][0]
    #             my_list.append(find)
    # for j in data_list:
    #     if j not in my_list:
    #         print(j)
    test_split()
    # nfi_img = nib.load('/home/amax/css/data/BHSD/data_for_Swin_UNETR/image/train/BHSD_image_000.nii.gz')
    # affine = nfi_img.affine
    # print(affine)
    pass
