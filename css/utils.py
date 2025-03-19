import os
os.chdir(os.getcwd())
import glob
import json
from pathlib import Path
import SimpleITK as sitk
# 关闭警告信息
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime
from scipy import stats
import shutil

# 修正后的全局样式配置（仅使用官方支持的rcParams）
STYLE_CONFIG = {
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'figure.dpi': 600,
    'savefig.bbox': 'tight',
    'hist.bins': 20
}

# 颜色和柱状图样式常量（作为普通变量使用）
BAR_STYLE = {
    'edgecolor': 'black',
    'linewidth': 0.5,
    'color_main': '#1f77b4',
    'color_accent': '#ff7f0e',
    'color_highlight': '#2ca02c'
}
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


def plot_train_test_distribution():
    # 数据集
    test_data = {'EDH':10, 'IPH':64, 'IVH':50, 'SAH':58, 'SDH':35}
    train_data = {'EDH':13, 'IPH':63, 'IVH':54, 'SAH':51, 'SDH':35}

    # 应用官方支持的rc配置
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # 使用独立的样式参数
    bar_width = 0.35
    index = np.arange(len(test_data))
    
    bars_test = ax.bar(index - bar_width/2, test_data.values(), bar_width,
                      color=BAR_STYLE['color_main'],
                      edgecolor=BAR_STYLE['edgecolor'],  # 直接指定
                      linewidth=BAR_STYLE['linewidth'],  # 直接指定
                      label='Test')
    
    bars_train = ax.bar(index + bar_width/2, train_data.values(), bar_width,
                       color=BAR_STYLE['color_accent'],
                       edgecolor=BAR_STYLE['edgecolor'],
                       linewidth=BAR_STYLE['linewidth'],
                       label='Train')
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.8,
                   f'{yval}', ha='center', va='bottom', 
                   fontsize=plt.rcParams['font.size']-3)
    
    add_labels(bars_test)
    add_labels(bars_train)

    # 坐标轴设置
    ax.set_xlabel('Hemorrhage Type')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Train-Test Distribution Comparison', 
                pad=20, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(test_data.keys())
    ax.set_ylim(0, max(max(test_data.values()), max(train_data.values()))*1.25)
    
    # 图例与边框
    ax.legend(frameon=True, loc='upper right', title='Dataset')
    ax.spines[['top', 'right']].set_visible(False)

    # 保存输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("BSHD_src_data", f"train_test_comparison_{timestamp}.png")
    plt.savefig(save_path, dpi=plt.rcParams['figure.dpi'])
    plt.close()

def plot_total_distribution():

    total_data = {'EDH':23, 'IPH':127, 'IVH':104, 'SAH':109, 'SDH':70}

    plt.rcParams.update(STYLE_CONFIG)
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    bars = ax.bar(total_data.keys(), total_data.values(),
                 width=0.3, 
                 color=BAR_STYLE['color_main'],
                 edgecolor=BAR_STYLE['edgecolor'],
                 linewidth=BAR_STYLE['linewidth'])
    ax.set_xlim(-0.5, len(total_data)-0.5)
    # 添加数值标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 3,
               f'{yval}', ha='center', va='bottom', 
               fontsize=plt.rcParams['font.size']-3)

    # 坐标轴设置
    ax.set_xlabel('Hemorrhage Type')
    ax.set_ylabel('Total Cases')
    ax.set_title('Total Case Distribution', 
                pad=20, fontweight='bold')
    ax.set_ylim(0, max(total_data.values())*1.15)
    
    # 边框设置
    ax.spines[['top', 'right']].set_visible(False)

    # 保存输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("BSHD_src_data", f"total_distribution_{timestamp}.png")
    plt.savefig(save_path, dpi=plt.rcParams['figure.dpi'])
    plt.close()


def select_slice(label_idx, data_path, save_path, image_base_path):
    """
    从标签文件中提取包含指定索引的切片，并在对应图像上高亮显示
    
    参数：
    label_idx: 要查找的标签值
    data_path: 标签文件目录路径
    save_path: 结果保存路径
    image_base_path: 图像文件基础路径
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 遍历所有标签文件
    for file_name in os.listdir(data_path):
        if file_name.endswith(('.nii', '.nii.gz')):
            file_path = os.path.join(data_path, file_name)
            
            try:
                # 加载标签数据
                label_img = nib.load(file_path)
                label_data = label_img.get_fdata()
                
                # 检查是否存在目标标签
                if np.any(label_data == label_idx):
                    # 构建对应的图像文件路径
                    image_path = os.path.join(
                        image_base_path,
                        f"BHSD_image_{file_name.split('_')[-1]}"
                    )
                    
                    # 加载对应的图像数据
                    if os.path.exists(image_path):
                        image_img = nib.load(image_path)
                        image_data = image_img.get_fdata()
                    else:
                        print(f"警告：对应的图像文件 {image_path} 不存在")
                        continue
                    
                    # 遍历所有切片
                    for z in range(label_data.shape[2]):
                        label_slice = label_data[:, :, z]
                        
                        # 检查当前切片是否包含目标标签
                        if np.any(label_slice == label_idx):
                            # 获取对应的图像切片
                            image_slice = image_data[:, :, z]
                            
                            # 预处理图像切片（归一化到0-255）
                            image_normalized = (image_slice - np.min(image_slice)) / (
                                np.max(image_slice) - np.min(image_slice)) * 255
                            image_8bit = image_normalized.astype(np.uint8)
                            
                            # 创建RGB图像
                            rgb_image = np.stack([image_8bit]*3, axis=-1)
                            
                            
                            # 创建标签掩膜并标记为红色
                            mask = label_slice == label_idx
                            rgb_image[mask] =  [255, 225, 0]
#                             colors = [
                                    #     [0, 184, 222],   # 医疗蓝（专业感）
                                    #     [123, 237, 159], # 薄荷绿（柔和清晰）
                                    #     [255, 131, 0],   # 警示橙（重点突出）
                                    #     [147, 112, 219], # 中紫色（良好对比）
                                    #     [255, 225, 0]    # 柠檬黄（高亮显示）
                                    # ]
                            # 保存图像
                            save_name = f"{os.path.splitext(file_name)[0]}_slice{z}.png"
                            save_file = os.path.join(save_path, save_name)
                            Image.fromarray(rgb_image).save(save_file)
                            
                            print(f"已保存：{save_file}")
                            
            except Exception as e:
                print(f"处理文件 {file_name} 时出错：{str(e)}")
                continue

                        

def cauculate_foreground_ratio():
    
    # 参数配置
    data_dir = "BSHD_src_data/label/total"  # 修改为实际路径
    output_dir = "BSHD_src_data/foriginal"  # 结果保存路径
    bins = 20
    dpi = 600
    save_formats = ['png', 'svg']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Date format without Chinese

    # Collect foreground ratios
    ratios = []
    for idx, file_path in enumerate(glob.glob(os.path.join(data_dir, "*.nii.gz"))):
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.uint8)
        total_voxels = data.size
        foreground = np.count_nonzero(data)
        ratio = (foreground / total_voxels) * 100 if total_voxels > 0 else 0
        ratios.append(ratio)
        print(f"Processed {idx+1} scans")

    ratios = np.array(ratios)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(ratios, 
                            bins=bins,
                            color='#1f77b4',
                            edgecolor='black',
                            alpha=0.8)

    # English labels
    plt.xlabel('Foreground Percentage (%)', fontsize=12)
    plt.ylabel('Number of CT Scans', fontsize=12)
    plt.title(f'Foreground Pixel Distribution (Total: {len(ratios)} Scans)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add count labels
    for patch in patches:
        x = patch.get_x() + patch.get_width()/2
        y = patch.get_height()
        if y > 0:
            plt.text(x, y, f'{int(y)}', 
                    ha='center', 
                    va='bottom',
                    fontsize=9)

    # Save outputs
    base_name = f"foreground_distribution_{timestamp}"
    for fmt in save_formats:
        save_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved {fmt} format: {save_path}")

    plt.close()

    # Save English statistics
    stats_path = os.path.join(output_dir, f"{base_name}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Data source: {data_dir}\n")
        f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Statistical Summary:\n")
        f.write(f"Total samples: {len(ratios)}\n")
        f.write(f"Min percentage: {np.min(ratios):.2f}%\n")
        f.write(f"Max percentage: {np.max(ratios):.2f}%\n")
        f.write(f"Mean percentage: {np.mean(ratios):.2f}%\n")
        f.write(f"Median percentage: {np.median(ratios):.2f}%\n")
        f.write(f"Standard deviation: {np.std(ratios):.2f}%\n")

    print(f"\nStatistics saved to: {stats_path}")


def cauculate_foreground_ratio_with_label():
    # 参数配置
    data_dir = "BSHD_src_data/label/total"  # 修改为实际路径
    result_dir = "BSHD_src_data/foriginal"  # 结果保存路径
    os.makedirs(result_dir, exist_ok=True)

    # 存储所有CT的前景比例
    foreground_ratios = []

    # 遍历处理每个CT文件
    for idx, file_path in enumerate(glob.glob(os.path.join(data_dir, "*.nii.gz"))):
        # 加载数据
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.uint8)
        
        # 计算前景比例（排除背景0）
        total_voxels = data.size
        foreground = np.count_nonzero(data)
        ratio = (foreground / total_voxels) * 100 if total_voxels > 0 else 0
        
        # 记录结果
        foreground_ratios.append(ratio)
        print(f"Processed {idx+1} scans | Current ratio: {ratio:.2f}%")

    # 转换为numpy数组
    ratios = np.array(foreground_ratios)

    # 创建带分布曲线的直方图
    plt.figure(figsize=(12, 6))

    # 直方图参数
    bins = 30
    hist_color = '#1f77b4'
    line_color = '#ff7f0e'

    # 绘制直方图
    n, bins, patches = plt.hist(ratios, bins=bins, density=True, 
                            alpha=0.7, color=hist_color,
                            edgecolor='black', linewidth=0.5)

    # 添加核密度估计曲线
    density = gaussian_kde(ratios)
    xs = np.linspace(0, ratios.max(), 200)
    plt.plot(xs, density(xs), color=line_color, lw=2, 
            label='Density Curve')

    # 添加统计信息
    mean = np.mean(ratios)
    median = np.median(ratios)
    plt.axvline(mean, color='red', linestyle='--', 
            linewidth=1.5, label=f'Mean: {mean:.2f}%')
    plt.axvline(median, color='green', linestyle='--',
            linewidth=1.5, label=f'Median: {median:.2f}%')

    # 图例和标签
    plt.xlabel('Foreground Percentage (%)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of Foreground Proportion in CT Scans\n'
            f'(Total {len(ratios)} Scans)', fontsize=14, pad=15)
    plt.legend()
    plt.grid(alpha=0.3)

    # 保存和显示
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'foreground_distribution_30bins.png'), dpi=600)
    plt.show()

    # 输出统计摘要
    print("\nStatistical Summary:")
    print(f"Scans Processed: {len(ratios)}")
    print(f"Mean ± Std: {np.mean(ratios):.2f}% ± {np.std(ratios):.2f}%")
    print(f"Range: [{np.min(ratios):.2f}% - {np.max(ratios):.2f}%]")
    print(f"25th-75th Percentile: {np.percentile(ratios,25):.2f}% - {np.percentile(ratios,75):.2f}%")



def get_data_min_max(data_path):
    data_min = []
    data_max = []
    for file_path in tqdm(glob.glob(os.path.join(data_path, "*.nii.gz"))):
        img = nib.load(file_path)
        data = img.get_fdata()
        data_min.append(np.min(data))
        data_max.append(np.max(data))
    print(f"min: {np.min(data_min)}")
    print(f"max: {np.max(data_max)}")

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob

# 配置参数
DATA_PATH = 'BSHD_src_data/image/total/'
FILE_PATTERN = '*.nii.gz'
OUTPUT_NAME = 'BSHD_src_data/pixelValue_Distribution.png'
MAX_BINS = 100                   # 减少分箱数量
AIR_THRESHOLD = -900
DISPLAY_MAX = 2000               # 显示范围上限
DPI = 600
MIN_PIXELS = 1000

def smart_bin_calculator(files, sample_num=3):
    """智能范围检测，自动适应显示范围"""
    min_val, max_val = np.inf, -np.inf
    valid_samples = 0
    
    for f in np.random.choice(files, min(sample_num, len(files)), replace=False):
        try:
            data = nib.load(f).get_fdata()
            masked_data = data[data > AIR_THRESHOLD]
            
            if masked_data.size == 0:
                continue
                
            # 应用显示范围过滤
            filtered_data = masked_data[masked_data <= DISPLAY_MAX]
            if filtered_data.size == 0:
                continue
                
            current_min = np.min(filtered_data)
            current_max = np.min([np.max(filtered_data), DISPLAY_MAX])
            
            min_val = min(min_val, current_min)
            max_val = max(max_val, current_max)
                
            valid_samples += 1
        except Exception as e:
            print(f"Error processing {f}: {str(e)}")
    
    if valid_samples == 0:
        raise ValueError("所有抽样文件无有效像素，请检查参数设置")
    
    # 保证最小范围
    if max_val - min_val < 100:
        max_val = min_val + 100
    
    return min_val, max_val

def pixel_value_distribution():
    # 文件处理
    all_files = glob.glob(os.path.join(DATA_PATH, FILE_PATTERN))
    
    if not all_files:
        raise FileNotFoundError(f"未找到匹配文件: {os.path.join(DATA_PATH, FILE_PATTERN)}")

    # 阶段1：动态范围检测
    try:
        global_min, global_max = smart_bin_calculator(all_files)
        print(f"有效值域范围: [{global_min:.1f}, {global_max:.1f}]")
        
        # 确保分箱合理性
        bin_step = max(10, (global_max - global_min)/MAX_BINS)
        bin_edges = np.arange(global_min, global_max + bin_step, bin_step)
        
    except ValueError as ve:
        print(ve)
        exit(1)

    # 阶段2：流式统计
    hist_counts = np.zeros(len(bin_edges)-1, dtype=np.int64)
    total_pixels = 0

    for fpath in tqdm(all_files, desc='Processing'):
        try:
            data = nib.load(fpath).get_fdata().flatten()
            # 同时应用阈值和显示范围过滤
            masked_data = data[(data > AIR_THRESHOLD) & (data <= DISPLAY_MAX)]
            
            if masked_data.size == 0:
                continue
                
            counts, _ = np.histogram(masked_data, bins=bin_edges)
            hist_counts += counts
            total_pixels += masked_data.size
            
        except Exception as e:
            print(f"\n处理 {fpath} 时出错: {str(e)}")
            continue

    # 有效性检查
    if total_pixels < MIN_PIXELS:
        raise RuntimeError(f"有效像素不足（{total_pixels}）")

    # 阶段3：统计计算
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # 均值计算
    mean_val = np.sum(bin_centers * hist_counts) / total_pixels
    
    # 中位数近似
    cumsum = np.cumsum(hist_counts)
    median_idx = np.searchsorted(cumsum, total_pixels // 2)
    median_approx = bin_centers[median_idx] if median_idx < len(bin_centers) else global_max

    # 阶段4：可视化
    plt.figure(figsize=(12, 6), dpi=DPI)
    
    # 绘制条形图
    bars = plt.bar(bin_centers, hist_counts / total_pixels, 
                 width=bin_width*0.8,
                 alpha=0.7, edgecolor='k', linewidth=0.5)
    
    # 添加统计线
    plt.axvline(mean_val, color='r', linestyle='--', lw=2, 
              label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_approx, color='g', linestyle='-.', lw=2,
              label=f'Median: {median_approx:.1f}')
    
    # 值域标注
    plt.text(0.98, 0.95, 
           f"Display Range: [-900, {DISPLAY_MAX}]",
           transform=plt.gca().transAxes,
           ha='right', va='top',
           bbox=dict(boxstyle='round', alpha=0.8, facecolor='w'))
    
    # 图形装饰
    plt.title(f'CT Pixel Distribution (n={len(all_files)}, {total_pixels/1e6:.1f}M pixels)', 
            pad=15, fontsize=12)
    plt.xlabel('Hounsfield Units (HU)', fontsize=10)
    plt.ylabel('Normalized Frequency', fontsize=10)
    plt.legend(fontsize=9)
    
    # 坐标轴设置
    plt.xlim(left=AIR_THRESHOLD, right=DISPLAY_MAX)
    plt.xticks(np.arange(AIR_THRESHOLD, DISPLAY_MAX+1, 200))
    plt.gca().ticklabel_format(useOffset=False, style='plain')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存输出
    plt.savefig(OUTPUT_NAME, bbox_inches='tight')
    print(f"\n结果已保存至 {OUTPUT_NAME}")

def Alignment():
    ref_data_path = 'BSHD_src_data/preprocessed_image/train/BHSD_image_000.nii.gz'
    tar_data_path = 'BSHD_src_data/preprocessed_image/train/BHSD_image_020.nii.gz'
    ref, tar = sitk.ReadImage(ref_data_path), sitk.ReadImage(tar_data_path)
    ref_origin, ref_Orientation = ref.GetOrigin(), ref.GetDirection()
    tar.SetOrigin(ref_origin)
    tar.SetDirection(ref_Orientation)
    sitk.WriteImage(tar, '/home/xiang/user/user_group/caoshangshang/RushBin/MONAI/BSHD_src_data/BHSD_image_020.nii.gz')


def change_spacing():
    # 输入和输出文件路径
    input_path = 'BSHD_src_data/preprocessed_image/train/BHSD_image_000.nii.gz'    # 输入文件路径，替换为你的输入文件
    output_path = '/home/xiang/user/user_group/caoshangshang/RushBin/MONAI/BSHD_src_data/BHSD_image_000.nii.gz' # 输出文件路径，替换为你的输出路径

    # 读取图像
    image = sitk.ReadImage(input_path)

    # 获取原始图像的spacing和尺寸
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    print(f"Original Spacing: {original_spacing}")
    print(f"Original Size: {original_size}")
    # 设置目标spacing值（根据需求修改）
    target_spacing = [original_spacing[0], original_spacing[1], 1.5]  # 示例：[x_spacing, y_spacing, z_spacing]

    # 计算调整spacing后的新尺寸（保持物理空间范围一致）
    new_size = [
        int(round(os * (osp / tsp)))
        for os, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    # 初始化ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)   # 设置目标spacing
    resampler.SetSize(new_size)                  # 设置新尺寸
    resampler.SetOutputDirection(image.GetDirection())  # 保持方向不变
    resampler.SetOutputOrigin(image.GetOrigin())        # 保持原点不变

    # 设置插值方法（根据数据类型选择）
    # sitk.sitkLinear - 适合连续数据（如CT/MRI）
    # sitk.sitkNearestNeighbor - 适合离散标签（如分割结果）
    resampler.SetInterpolator(sitk.sitkLinear)

    # 执行重采样
    resampled_image = resampler.Execute(image)

    # 保存结果
    sitk.WriteImage(resampled_image, output_path)

    # 验证输出参数
    print(f"New Spacing: {resampled_image.GetSpacing()}")
    print(f"New Size: {resampled_image.GetSize()}")


def get_label_with_EDH_scan():
    # ['097' '117' '119' '122' '124' '133' '136' '144' '149' '153' '163' '186' '190']
    path = 'BSHD_src_data/test_slice/test_slice_1'
    scan_list = []
    for img in tqdm(glob.glob(path+'/*.png')):
        scan_list.append(img.split('.')[0].split('_')[-1])
    scan_list = np.unique(scan_list)
    print(scan_list)


def mv_log2file():
    log_list = glob.glob('css/*.log')
    file_list = os.listdir('css/experiment/swim_unetr')
    file_dirname = 'css/experiment/swim_unetr/'
    if len(log_list) == 0:
        print('no log file')
        exit()
    for log in log_list:
        log_name = os.path.basename(log).split('.')[0]
        if log_name in file_list:
            try:
                shutil.move(log, file_dirname+f'{log_name}')
            except:
                print(f'{log_name} move failed')

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
    # 添加数值标签
    # select_slice(
    #     label_idx=5,  # 要查找的标签值
    #     data_path="BSHD_src_data/label/test",  # 标签文件目录
    #     save_path="BSHD_src_data/test_slice/test_slice_5",  # 结果保存目录
    #     image_base_path="BSHD_src_data/preprocessed_image/test"  # 图像文件基础目录
    # )
    # cauculate_foreground_ratio()
    # 执行绘图
    # plot_train_test_distribution()
    # plot_total_distribution()
    # cauculate_foreground_ratio_with_label()
    # get_data_min_max(data_path='BSHD_src_data/preprocessed_image/test')
    # pixel_value_distribution()
    # Alignment()
    # change_spacing()
    # get_label_with_EDH_scan()
    mv_log2file()
    pass




    