import os
os.chdir(os.getcwd())
import glob
import json
from pathlib import Path
import SimpleITK as sitk
# 关闭警告信息
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from tqdm import tqdm
import nibabel as nib
def generate_data_list(img_path, label_path, save_path=''):
    img_list = glob.glob(f'{img_path}/*.nii.gz')
    label_list = glob.glob(f'{label_path}/*.nii.gz')
    assert len(img_list) == len(label_list), f'num of img is different with num of label!'
    data_list = []
    for img in img_list:
        label = os.path.join(label_path, 'BHSD_label_' + os.path.basename(img).split('_')[-1])
        assert os.path.isfile(label), f'{label} not exsit!'
        data_list.append({"image":img, "label":label})
    print(f'totally {len(data_list)} cases!')
    # with open(save_path, 'w') as f:
    #     json.dump(data_list, f, indent=4)
    return data_list

def get_data_info(data_path, save_path):
    """
    get origin, spacing, pixel value(min,max)
    """
    data_path_list = glob.glob(f'{data_path}/*.nii.gz')
    data_info = {}
    pixel_value = {'min':0., 'max':0.}
    all_pixel_values = []
    print('Loading data ...')
    for data in tqdm(data_path_list):
        data_name = os.path.basename(data)
        img = sitk.ReadImage(data)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()
        np_data = sitk.GetArrayFromImage(img)
        # all_pixel_values.extend(np_data.flatten())
        min_value = np.min(np_data)
        max_value = np.max(np_data)
        all_pixel_values.extend([min_value, max_value])
        
        if min_value < pixel_value['min']:
            pixel_value['min'] = min_value
        if max_value > pixel_value['max']:
            pixel_value['max'] = max_value    
        data_info[data_name] = {'spacing': tuple(spacing),
                                'origin': tuple(origin),
                                'direction': tuple(direction),
                                'min_value': float(min_value),
                                'max_value': float(max_value)}
        
    print('Finished!')
    # 绘制合并直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_pixel_values, bins='auto', color='blue', alpha=0.7)
    plt.title("Pixel Distribution Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    # 使用MaxNLocator自动找到最佳的刻度位置
    ax = plt.gca()  # 获取当前轴
    ax.xaxis.set_major_locator(MaxNLocator(100))  # 设置x轴的主要刻度显示最多10个刻度
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(os.path.dirname(save_path), 'process_hist_train.png'), dpi=300)
    plt.close()  # 关闭图像
    
    with open(save_path, 'w') as f:
        json.dump(data_info, f, indent=4)
    print(f"min_pixel_value:{pixel_value['min']}, min_pixel_value:{pixel_value['max']}")


def clip_and_normalize(volume, min_val=-40, max_val=120):
    # 将图像裁剪到指定范围内
    volume = np.clip(volume, min_val, max_val)
    # 将图像标准化到 [0, 1] 范围内
    volume = (volume - 40) / 80
    return volume


def clip_norm_data(data_path, save_path):
    data_list = glob.glob(f'{data_path}/*.nii.gz')
    data_list = tqdm(data_list, desc=f'processing: x / 96')
    for i, data in enumerate(data_list):
        i = i + 1
        data_list.set_description(f'processing: {i} / 96')
        name = os.path.basename(data)
        img = sitk.ReadImage(data)
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
    


import os
import numpy as np
import nibabel as nib

def count_cases_with_labels(data_folder, label_values):

    label_presence = {label: 0 for label in label_values}
    
    # 遍历文件夹中的所有.nii或.nii.gz文件
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            file_path = os.path.join(data_folder, file_name)
            img = nib.load(file_path)  # 加载NIfTI图像
            data = img.get_fdata()     # 获取图像数据作为numpy数组
            
            # 检查当前文件是否包含任何需要统计的标签
            for label in label_values:
                if np.any(data == label):  # 如果该文件至少有一个像素属于此标签，则增加计数
                    label_presence[label] += 1
                
    return label_presence




        




if __name__ == '__main__':
    # generate_data_list('BSHD_src_data/image/test',
    #                     'BSHD_src_data/label/test',
    #                     'BSHD_src_data/test.json')

    # get_data_info('BSHD_src_data/preprocessed_image/train', 'BSHD_src_data/test_data_info.json')
    # clip_norm_data(data_path='./BSHD_src_data/image/test',
                    # save_path='./BSHD_src_data/preprocessed_image/test')
                    
    # data_folder = 'BSHD_src_data/label/test'  # 替换为你的数据集目录
    # label_values = [1., 2., 3., 4., 5.]  # 假设这五个数字是你五种前景标签的值
    # presence_counts = count_cases_with_labels(data_folder, label_values)

    # print("Cases with labels:", presence_counts)
    
    pass




    