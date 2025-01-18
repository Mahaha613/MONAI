import os
import shutil
from glob import glob


def generate_keys_for_sort(pth_file):
    file_name = os.path.basename(pth_file).split('.pth')[0]
    name = float(file_name.split(':')[-1])
    return name

def save_memery(path):
    exp_folder = os.listdir(path)
    for exp in exp_folder:
        if len(os.listdir(os.path.join(path, exp))) > 6:
            print(exp)
            pth_files = sorted(glob(os.path.join(path, exp) + '/*.pth'), key=generate_keys_for_sort)[:-3]
            with open('/home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/experiment/rm.log', 'a') as f:
                for pth in pth_files:
                    f.write(pth + '\n')
            # print(pth_files)
            # for pth in pth_files:
            #     os.remove(pth)





if __name__ == '__main__':
    path = '/home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/experiment/swim_unetr'
    save_memery(path)
