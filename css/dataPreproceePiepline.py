import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载NIfTI数据
def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

# 数据截断函数：根据给定的窗宽和窗位调整图像数据
def apply_window(image, window_level, window_width):
    image_min = window_level - (window_width / 2.0)
    image_max = window_level + (window_width / 2.0)
    image_clipped = np.clip(image, image_min, image_max)
    image_windowed = (image_clipped - image_min) / window_width
    image_normalized_m1_1 = image_windowed * 2 - 1
    return image_windowed

# 显示一组切片（原始、处理后、标签），并将它们放在一张图上
def show_and_save_slices_group(data, label, slices, save_path, window_level=40, window_width=160):
    num_slices = len(slices)
    fig, axes = plt.subplots(3, num_slices, figsize=(5 * num_slices, 15))
    
    for i, slice_idx in enumerate(slices):
        # 原始切片
        axes[0, i].imshow(data[:, :, slice_idx], cmap="gray", origin="lower")
        axes[0, i].set_title(f'Original Slice {slice_idx}')
        
        # 处理后的切片
        adjusted_slice = apply_window(data[:, :, slice_idx], window_level, window_width)
        axes[1, i].imshow(adjusted_slice, cmap="gray", origin="lower")
        axes[1, i].set_title(f'Windowed-norm Slice {slice_idx}')
        
        # 标签
        axes[2, i].imshow(label[:, :, slice_idx], cmap="gray", origin="lower")
        axes[2, i].set_title(f'Label Slice {slice_idx}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def get_label_slice_num(label_path):
    print(f'name: {os.path.basename(label_path)}\n')
    label = load_nifti(label_path)
    print(f'shape:{label.shape}\n')
    slice_with_label = []
    for slice in range(label.shape[2]):
        if len(np.unique(label[:, :, slice])) > 1:
            slice_with_label.append(slice)
            print(f'slice_label:{slice}\n')
    if len(slice_with_label) > 3:
        slice_with_label = slice_with_label[:3]
    return slice_with_label

def main(dataPath, labelPath, savePath):
    assert os.path.exists(dataPath) and os.path.exists(labelPath), f'{dataPath} or {labelPath} is not exists'
    assert (os.path.basename(dataPath)).split('_')[-1] == os.path.basename(labelPath).split('_')[-1], f'{dataPath} and {labelPath} is not pair data'
    
    slice_with_label = get_label_slice_num(labelPath)
    data = load_nifti(dataPath)
    label = load_nifti(labelPath)
    
    # 选择前4个带有标签的切片展示
    selected_slices = slice_with_label[:4]
    show_and_save_slices_group(data, label, selected_slices, savePath)

if __name__ == '__main__':
    dataPath = 'BSHD_src_data/image/train/BHSD_image_010.nii.gz'
    labelPath = 'BSHD_src_data/label/train/BHSD_label_010.nii.gz'
    savePath = 'css/window_norm.png'  # 保存路径
    main(dataPath, labelPath, savePath)