import os
os.chdir(os.getcwd())
import glob
import json
from pathlib import Path


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




if __name__ == '__main__':
    generate_data_list('BSHD_src_data/image/test',
                        'BSHD_src_data/label/test',
                        'BSHD_src_data/test.json')




    