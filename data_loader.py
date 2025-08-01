# Imports
import os
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy import signal
from PIL import Image
import pandas as pd
import torch.utils.data as Data
import numpy as np
import scipy.io as sci
import os.path as osp
from EEG_reshape import EEG_reshape,read_groundtruth,quaternion_to_rotation_matrix,find_closest_timestamp
import cv2
from read_pose import read_pose
import scipy.io as sio 
import re 
from utils_ import  quaternion_angular_error,process_poses,load_state_dict,load_image,qlog

import random

def image_load(image_path, image_name):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_file = os.path.join(image_path, image_name)
    image = Image.open(image_file).convert('RGB')
    image = transform(image)
    return image

def image_loader(image_path):
    image_dict = {}
    for image_name in os.listdir(image_path):
        image = image_load(image_path, image_name)
        image_dict[image_name[:-5]] = image
    return image_dict

# Dataset class
class EEGDataset:
    # Constructor
    def __init__(self, opt):
        # Load EEG signals
        self.opt = opt
        loaded = torch.load(self.opt.eeg_dataset)
        self.image_dict = image_loader(opt.image_dataset)
        if opt.subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        else:
            self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.opt.time_low:self.opt.time_high,:]
        if self.opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1,128,self.opt.time_high-self.opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Get image
        image_name = self.images[self.data[i]["image"]]
        image = self.image_dict[image_name]
        # Return
        return eeg, label, image

# Splitter class
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data                                                                                   
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        HZ = 1000
        low_f, high_f = 5, 95
        b, a = signal.butter(2, [low_f*2/HZ, high_f*2/HZ], 'bandpass')
        for i in self.split_idx:
            self.dataset.data[i]["eeg"] = torch.from_numpy(signal.lfilter(b, a, self.dataset.data[i]["eeg"]).copy())
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label, image = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label, image


def data_loader1(opt):    #data for-eXp1
    sample_size = 300
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size / 3)]
    test_indices = indices[-int(sample_size / 3):]
    np.savetxt(' /test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt(' /test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 500
    channels = 1
    #随机数据
    # import re  
    # # 打开文件并读取内容  
    # with open('/root/autodl-tmp/基于空间想象的脑电信号处理/trail5.txt', 'r') as file:  
    #     lines = file.readlines()  
    # # 使用正则表达式提取括号中的数字  
    # numbers = [re.search(r'\((\d+)\)', line).group(1) for line in lines]  
    # train_indices =  numbers[:-int(sample_size / 3)]
    # test_indices =  numbers[-int(sample_size / 3):]
    print(train_indices)
    print(test_indices)
    n_step = 60
    n_input = 500
    channels = 1

    
    EEG_path=" /Scene/NP/order7.mat"
    #" /Scene/order2.mat"
#" /Scene/NP/Scene1.mat"
    if     EEG_path==" /Scene/order1.mat":
        grouthtruth_txt1=' rgbd_dataset_freiburg1_desk/groundtruth.txt'
        rgb_txt1=" rgbd_dataset_freiburg1_desk/rgb.txt"
    if     EEG_path==" /Scene/order2.mat":
        grouthtruth_txt1=' rgbd_dataset_freiburg1_desk2/groundtruth.txt'
        rgb_txt1=" rgbd_dataset_freiburg1_desk2/rgb.txt"

    if     EEG_path==" /Scene/order3.mat":
        grouthtruth_txt1=' scene3_long_office_household/groundtruth.txt'
        rgb_txt1=" scene3_long_office_household/rgb.txt"

    if     EEG_path==" /Scene/order4.mat":
        grouthtruth_txt1=' scene4_desk/groundtruth.txt'
        rgb_txt1=" scene4_desk/rgb.txt"

    if     EEG_path==" /Scene/order5.mat":
        grouthtruth_txt1=' scene5_plant/groundtruth.txt'
        rgb_txt1=" scene5_plant/rgb.txt"

    if     EEG_path==" /Scene/order6.mat":
        grouthtruth_txt1=' scene6_teddy/groundtruth.txt'
        rgb_txt1=" scene6_teddy/rgb.txt"

    if     EEG_path==" /Scene/order7.mat":
        grouthtruth_txt1='/ scene7_floor/groundtruth.txt'
        rgb_txt1="/ /scene7_floor/rgb.txt"




    if     EEG_path==" /Scene/NP/order1.mat":
        grouthtruth_txt1=' rgbd_dataset_freiburg1_desk/groundtruth.txt'
        rgb_txt1=" rgbd_dataset_freiburg1_desk/rgb.txt"
    if     EEG_path==" /Scene/NP/order2.mat":
        grouthtruth_txt1=' rgbd_dataset_freiburg1_desk2/groundtruth.txt'
        rgb_txt1=" rgbd_dataset_freiburg1_desk2/rgb.txt"

    if     EEG_path==" /Scene/NP/order3.mat":
        grouthtruth_txt1=' scene3_long_office_household/groundtruth.txt'
        rgb_txt1=" scene3_long_office_household/rgb.txt"

    if     EEG_path==" /Scene/NP/order4.mat":
        grouthtruth_txt1=' scene4_desk/groundtruth.txt'
        rgb_txt1=" scene4_desk/rgb.txt"

    if     EEG_path==" /Scene/NP/order5.mat":
        grouthtruth_txt1=' scene5_plant/groundtruth.txt'
        rgb_txt1=" scene5_plant/rgb.txt"

    if     EEG_path==" /Scene/NP/order6.mat":
        grouthtruth_txt1=' scene6_teddy/groundtruth.txt'
        rgb_txt1=" scene6_teddy/rgb.txt"

    if     EEG_path==" /Scene/NP/order7.mat":
        grouthtruth_txt1='/ scene7_floor/groundtruth.txt'
        rgb_txt1="/ /scene7_floor/rgb.txt"





    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt1,rgb_txt1,EEG=True)
    eeg_y_test=[]
    eeg_y_train=[]
    from scipy.spatial.transform import Rotation as R

    save_dir = ' /test/BCML/memory_trail'
    os.makedirs(save_dir, exist_ok=True)
    
    # 自适应命名：从EEG_path中提取文件名并生成相应的trail名称
    import os as path_os
    eeg_filename = path_os.path.basename(EEG_path).replace('.mat', '')  # 提取文件名，去掉扩展名
    
    # 根据路径判断是否为NP版本
    if '/NP/' in EEG_path:
        trail_name = f"trail_{eeg_filename}_NP.txt"
    else:
        trail_name = f"trail_{eeg_filename}.txt"
    
    save_path = path_os.path.join(save_dir, trail_name)

    # 计算
    eeg_poses = np.array(eeg_poses[:500])
    positions = eeg_poses[:, :3]
    trans_deltas = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    total_trans = np.sum(trans_deltas)

    quaternions = eeg_poses[:, 3:]
    rot1 = R.from_quat(quaternions[1:])
    rot0 = R.from_quat(quaternions[:-1])
    relative_rot = rot1 * rot0.inv()
    rot_deltas = np.rad2deg(relative_rot.magnitude())
    total_rot = np.sum(rot_deltas)

    # 输出文本
    with open(save_path, 'w') as f:
        f.write(f"EEG文件: {EEG_path}\n")
        f.write(f"Ground Truth: {grouthtruth_txt1}\n")
        f.write(f"RGB文件: {rgb_txt1}\n")
        f.write(f"样本数量: {len(eeg_poses)}\n")
        f.write(f"平均位移变化: {np.mean(trans_deltas):.4f} m\n")
        f.write(f"总位移距离: {total_trans:.2f} m\n")
        f.write(f"平均旋转变化: {np.mean(rot_deltas):.2f}°\n")
        f.write(f"总角度变化: {total_rot:.2f}°\n")

    print(f"结果已保存到：{save_path}")
    
    
    eeg_x_test = eeg_data[test_indices]
    #eeg_x_test = eeg_data[-60:,:,:]

    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]
    #eeg_x_train = eeg_data[:120,:,:]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    #data_in = pd.read_csv(opt.facial_image_dataset + 'image_1_32.csv', header=None)
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(300)
    #frame_idx=numbers  
    if   EEG_path==" /Scene/order1.mat" or EEG_path==" /Scene/NP/order1.mat":
        c_imgs = [osp.join(' /Scene/order1/', '1_test({:d}).png'.format(int(i+1)))for i in frame_idx]
    if   EEG_path==" /Scene/order2.mat" or EEG_path==" /Scene/NP/order2.mat" :
        c_imgs = [osp.join(' /Scene/order2/', '2_test ({:d}).png'.format(int(i+1)))for i in frame_idx]
    if   EEG_path==" /Scene/order3.mat" or EEG_path==" /Scene/NP/order3.mat":
        c_imgs = [osp.join(' /test/BCML/data/Work/Scene3_office/', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    if   EEG_path==" /Scene/order4.mat" or EEG_path==" /Scene/NP/order4.mat":
        c_imgs = [osp.join(' /test/BCML/data/Work/Scene4_desk/', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    if   EEG_path==" /Scene/order5.mat" or EEG_path==" /Scene/NP/order5.mat":
        c_imgs = [osp.join(' /test/BCML/data/Work/Scene5_plant/', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    if   EEG_path==" /Scene/order6.mat" or EEG_path==" /Scene/NP/order6.mat":
        c_imgs = [osp.join(' /test/BCML/data/Work/Scene6_teddy/', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]
    if   EEG_path==" /Scene/order7.mat" or EEG_path==" /Scene/NP/order7.mat":
        c_imgs = [osp.join(' /test/BCML/data/Work/Scene7_floor/', 'frame-{:05d}.color.png'.format(int(i)))for i in frame_idx]


    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)
    image_poses = read_groundtruth(grouthtruth_txt1,rgb_txt1,EEG=False)

    image_x_test=image_data[test_indices]
    #image_x_test = image_data[-60:,:]

    for i in test_indices:
        i=int(i)
        image_y_test.append(image_poses[i])



    image_x_train=image_data[train_indices]
    #image_x_train = image_data[:120,:]

    for i in train_indices:
        i=int(i)
        image_y_train.append(image_poses[i])






    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')
    #eeg_y_train = eeg_y_train.astype('float32')
    #eeg_y_test = eeg_y_test.astype('float32')
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')
    image_x_train = image_x_train.transpose(0, 3, 1, 2)
    image_x_test = image_x_test.transpose(0, 3, 1, 2)      
    image_x_train = image_x_train / 255.0
    image_x_test = image_x_test / 255.0
    placeholder_image = np.zeros_like(image_x_train[0])

    # # ——训练集中随机 30% 图像使用占位符——
    # placeholder_ratio = 0.3
    # num_placeholders = int(len(image_x_train) * placeholder_ratio)
    # placeholder_indices = np.random.choice(len(image_x_train), num_placeholders, replace=False)
    # for idx in placeholder_indices:
    #     image_x_train[idx] = placeholder_image

    # ——测试集图像全部使用占位符——
    image_x_test = np.tile(placeholder_image, (len(image_x_test), 1, 1, 1))

    #image_y_train = image_y_train.astype('float32')
    #image_y_test = image_y_test.astype('float32')
    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)

    train_dataset = Data.TensorDataset(torch.tensor(eeg_x_train).unsqueeze(1), torch.tensor(image_y_train), torch.tensor(image_x_train))
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = Data.TensorDataset(torch.tensor(eeg_x_test).unsqueeze(1), torch.tensor(image_y_test), torch.tensor(image_x_test))
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if opt.mode == 'train':
        return_dataloader = train_dataloader
    elif opt.mode == 'val':
        return_dataloader = test_dataloader
    else:
        return_dataloader = test_dataloader

    return train_dataloader, test_dataloader

def data_loader2(opt):      # data for-eXp2
                         
    sample_size = 500
    #顺序数据
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    np.savetxt(' /test/BCML/data/Work/SPLIT/train_indices.csv', train_indices, fmt='%d')
    np.savetxt(' /test/BCML/data/Work/SPLIT/test_indices.csv', test_indices, fmt='%d')
    train_indices = np.loadtxt(opt.facial_splits_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(opt.facial_splits_path + 'test_indices.csv').astype('int')
    n_step = 60
    n_input = 500
    channels = 1

    print(train_indices)

    EEG_path=" /实验五/Subject05/NP/sub05_seq_500.mat"
    #" /subject2/实验1/order1.mat"
    
    grouthtruth_txt=' rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt=" rgbd_dataset_freiburg1_desk/rgb.txt"

    print(sio.loadmat(EEG_path))
    eeg_data = EEG_reshape(EEG_path)
    eeg_poses = read_groundtruth(grouthtruth_txt,rgb_txt,EEG=True)
    eeg_y_test=[]
    eeg_y_train=[]
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i=int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]

    for i in train_indices:
        i=int(i)
        eeg_y_train.append(eeg_poses[i])

    height = 256
    width = 256
    channels = 3
    image_data=[]
    image_y_test=[]
    image_y_train=[]

    frame_idx=range(500)
    #frame_idx=numbers  

    c_imgs = [osp.join(' rgbd_dataset_freiburg1_desk/rgb', 'test1({:d}).png'.format(int(i+1)))for i in frame_idx]
    #/root/autodl-tmp/test/BCML/data/Work/desk2
    for img_path in c_imgs:  
        img = cv2.imread(img_path)  # 默认读取彩色图像
        if img is None:
            print("?")  
        if img is not None:  
            img= cv2.resize(img, (width, height))  
            img_new = np.expand_dims(img, axis=-1)  # 增加一个通道维度 
            img_flat = img_new.flatten() 
            image_data.append(img_flat)  
        
    image_data = np.array(image_data)

    image_poses = read_groundtruth(grouthtruth_txt, rgb_txt, EEG=False)
    image_x_test = image_data[test_indices]
    image_y_test = [image_poses[int(i)] for i in test_indices]
    image_x_train = image_data[train_indices]
    image_y_train = [image_poses[int(i)] for i in train_indices]

    # EEG reshape
    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')

    # Image reshape
    image_x_train = image_x_train.reshape(-1, height, width, channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, channels).astype('float32')

    # 转换为 NCHW
    image_x_train = image_x_train.transpose(0, 3, 1, 2) / 255.0
    image_x_test = image_x_test.transpose(0, 3, 1, 2) / 255.0

    # 设置占位符图像
    placeholder_image = np.zeros_like(image_x_train[0])

    # # ——训练集中随机 30% 图像使用占位符——
    # placeholder_ratio = 0.3
    # num_placeholders = int(len(image_x_train) * placeholder_ratio)
    # placeholder_indices = np.random.choice(len(image_x_train), num_placeholders, replace=False)
    # for idx in placeholder_indices:
    #     image_x_train[idx] = placeholder_image

    # ——测试集图像全部使用占位符——
    image_x_test = np.tile(placeholder_image, (len(image_x_test), 1, 1, 1))

    # 转为 Tensor 并构建 DataLoader
    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)

    train_dataset = Data.TensorDataset(
        torch.tensor(eeg_x_train).unsqueeze(1),
        torch.tensor(image_y_train),
        torch.tensor(image_x_train)
    )
    test_dataset = Data.TensorDataset(
        torch.tensor(eeg_x_test).unsqueeze(1),
        torch.tensor(image_y_test),
        torch.tensor(image_x_test)
    )

    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader

def data_loader3(opt, brain_region='all'):         #data for-eXp4
    """
    脑区通道划分数据加载器 - 基于原data_loader61修改
    
    Args:
        opt: 配置参数
        brain_region: 脑区选择，可选 'frontal', 'parietal', 'occipital', 'temporal', 'central', 'all'
    """
    
    brain_region_channels = {
        'frontal': {  # 额叶区
            'channels': ['FP1', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8',
                        'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6'],
            'indices': [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21],  # 对应电极索引
            'name': '额叶区'
        },
        'parietal': {  # 顶叶区
            'channels': [
                'C5', 'C3', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6',
                'P5', 'P3', 'P1', 'P2', 'P4', 'P6'
            ],
            'indices': [24, 25, 29, 30, 33, 34, 35, 37, 38, 39, 42, 43, 44, 46, 47, 48],  # 对应电极索引
            'name': '顶叶区'
        },
        'occipital': {  # 枕叶区
            'channels': ['PO7', 'PO5', 'PO3', 'PO4', 'PO6', 'PO8', 'O1', 'O2'],
            'indices': [50, 51, 52, 54, 55, 56, 57, 59],  # 对应电极索引
            'name': '枕叶区'
        },
        'temporal': {  # 颞叶区
            'channels': ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8'],
            'indices': [14, 22, 23, 31, 32, 40, 41, 49],  # 对应电极索引
            'name': '颞叶区'
        },
        'left_hemisphere': {  # 左半球
            'channels': ['FP1', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FC5', 'FC3', 'FC1',
                        'C5', 'C3', 'FT7', 'T7', 'TP7',
                        'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1',
                        'PO7', 'PO5', 'PO3', 'O1'],
            'indices': [0, 3, 5, 6, 7, 8, 15, 16, 17, 24, 25, 14, 23, 32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 57],
            'name': '左半球'
        },
        'right_hemisphere': {  # 右半球
            'channels': ['FP2', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FC6', 'FC4', 'FC2',
                        'C6', 'C4', 'FT8', 'T8', 'TP8',
                        'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2',
                        'PO8', 'PO6', 'PO4', 'O2'],
            'indices': [2, 4, 13, 12, 11, 10, 21, 20, 19, 30, 29, 22, 31, 40, 39, 38, 37, 49, 48, 47, 46, 56, 55, 54, 59],
            'name': '右半球'
        }}
        
    if brain_region not in brain_region_channels:
        raise ValueError(f"不支持的脑区: {brain_region}. 支持的脑区: {list(brain_region_channels.keys())}")
    
    # 获取当前脑区的通道信息
    selected_info = brain_region_channels[brain_region]
    selected_channels = selected_info['channels']
    channel_indices = selected_info['indices']
    region_name = selected_info['name']
    
    print(f"\n=== {region_name} 实验开始 ===")
    print(f"选择的通道: {selected_channels}")
    print(f"通道数量: {len(selected_channels) if brain_region != 'all' else '60 (全部通道)'}")
    print(f"通道索引: {channel_indices if len(channel_indices) <= 20 else str(channel_indices[:10]) + '...(显示前10个)'}")
    
    sample_size = 500
    
    # 数据划分 - 保持与原函数一致
    indices = np.arange(sample_size)
    np.random.shuffle(indices)
    train_indices = indices[:-int(sample_size // 3)]
    test_indices = indices[-int(sample_size // 3):]
    
    # 为不同脑区创建独立的保存路径
    split_base_path = ' /test/BCML/data/Work/SPLIT/'
    region_split_path = split_base_path + f'{brain_region}/'
    os.makedirs(region_split_path, exist_ok=True)
    
    np.savetxt(region_split_path + 'train_indices.csv', train_indices, fmt='%d')
    np.savetxt(region_split_path + 'test_indices.csv', test_indices, fmt='%d')
    
    # 加载保存的划分索引
    train_indices = np.loadtxt(region_split_path + 'train_indices.csv').astype('int')
    test_indices = np.loadtxt(region_split_path + 'test_indices.csv').astype('int')
    
    # 网络参数设置
    n_input = 500
    if brain_region == 'all':
        n_step = 60  # 60个通道
    else:
        n_step = len(channel_indices)  # 根据选择的脑区通道数调整
    channels = 1

    print(f"训练索引示例: {train_indices[:5]}...")

    # 数据路径 - 保持与原函数一致
    EEG_path = " /实验五/Subject05/NP/sub05_seq_500.mat"
    grouthtruth_txt = ' rgbd_dataset_freiburg1_desk/groundtruth.txt'
    rgb_txt = " rgbd_dataset_freiburg1_desk/rgb.txt"

    print("正在加载EEG数据...")
    print(sio.loadmat(EEG_path))
    
    # 获取原始EEG数据
    raw_eeg_data = EEG_reshape(EEG_path)
    
    # 根据脑区选择进行通道筛选
    if brain_region != 'all':
        # 根据原始EEG数据的维度进行处理

        if len(raw_eeg_data.shape) == 3:
            # 情况3: 数据格式已经是(samples, timesteps, channels) = (500, 60, 60)
            if raw_eeg_data.shape[1] == 60:
                # 直接选择通道
                eeg_data = raw_eeg_data[:, channel_indices,:]
                # flatten为(samples, timesteps*selected_channels)
                eeg_data = eeg_data.reshape(sample_size, -1)
            else:
                print(f"警告: 通道数不匹配 {raw_eeg_data.shape}")
                eeg_data = raw_eeg_data.reshape(sample_size, -1)
        else:
            print(f"警告: 不支持的数据维度 {raw_eeg_data.shape}")
            eeg_data = raw_eeg_data
            
        print(f"EEG数据维度调整: {raw_eeg_data.shape} -> {eeg_data.shape}")
        print(f"选择的通道索引: {channel_indices}")
        print(f"对应的通道名称: {selected_channels}")
    else:
        eeg_data = raw_eeg_data
        print(f"使用完整EEG数据: {eeg_data.shape} (全部60通道)")
    
    # EEG姿态数据处理
    eeg_poses = read_groundtruth(grouthtruth_txt, rgb_txt, EEG=True)
    eeg_y_test = []
    eeg_y_train = []
    eeg_x_test = eeg_data[test_indices]
    for i in test_indices:
        i = int(i)
        eeg_y_test.append(eeg_poses[i])

    eeg_x_train = eeg_data[train_indices]
    for i in train_indices:
        i = int(i)
        eeg_y_train.append(eeg_poses[i])

    # 图像数据处理 - 保持与原函数完全一致
    height = 256
    width = 256
    image_channels = 3
    image_data = []
    image_y_test = []
    image_y_train = []

    frame_idx = range(500)
    c_imgs = [osp.join(' rgbd_dataset_freiburg1_desk/rgb', 'test1({:d}).png'.format(int(i+1))) 
              for i in frame_idx]

    print("正在加载图像数据...")
    for img_path in c_imgs:
        img = cv2.imread(img_path)
        if img is None:
            print("图像读取失败")
        if img is not None:
            img = cv2.resize(img, (width, height))
            img_new = np.expand_dims(img, axis=-1)
            img_flat = img_new.flatten()
            image_data.append(img_flat)
        
    image_data = np.array(image_data)

    image_poses = read_groundtruth(grouthtruth_txt, rgb_txt, EEG=False)
    image_x_test = image_data[test_indices]
    image_y_test = [image_poses[int(i)] for i in test_indices]
    image_x_train = image_data[train_indices]
    image_y_train = [image_poses[int(i)] for i in train_indices]

    # EEG数据重塑
    eeg_x_train = eeg_x_train.reshape(-1, n_step, n_input).astype('float32')
    eeg_x_test = eeg_x_test.reshape(-1, n_step, n_input).astype('float32')

    # 图像数据重塑
    image_x_train = image_x_train.reshape(-1, height, width, image_channels).astype('float32')
    image_x_test = image_x_test.reshape(-1, height, width, image_channels).astype('float32')

    # 转换为 NCHW
    image_x_train = image_x_train.transpose(0, 3, 1, 2) / 255.0
    image_x_test = image_x_test.transpose(0, 3, 1, 2) / 255.0

    # 设置占位符图像
    placeholder_image = np.zeros_like(image_x_train[0])

    # 测试集图像全部使用占位符 - 保持原有逻辑
    image_x_test = np.tile(placeholder_image, (len(image_x_test), 1, 1, 1))

    # 转为 Tensor 并构建 DataLoader
    eeg_x_train = np.array(eeg_x_train)
    image_y_train = np.array(image_y_train)
    image_x_train = np.array(image_x_train)
    eeg_x_test = np.array(eeg_x_test)
    image_y_test = np.array(image_y_test)
    image_x_test = np.array(image_x_test)

    train_dataset = Data.TensorDataset(
        torch.tensor(eeg_x_train).unsqueeze(1),
        torch.tensor(image_y_train),
        torch.tensor(image_x_train)
    )
    test_dataset = Data.TensorDataset(
        torch.tensor(eeg_x_test).unsqueeze(1),
        torch.tensor(image_y_test),
        torch.tensor(image_x_test)
    )

    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 打印实验统计信息
    print(f"\n=== {region_name} 数据加载完成 ===")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"EEG数据形状: train={eeg_x_train.shape}, test={eeg_x_test.shape}")
    print(f"图像数据形状: train={image_x_train.shape}, test={image_x_test.shape}")
    print(f"网络输入维度: n_step={n_step}, n_input={n_input}")
    if brain_region != 'all':
        print(f"选择的通道索引: {channel_indices}")
    print("="*50)

    return train_dataloader, test_dataloader


def get_loader(opt):
    # Load DataLoader of given DialogDataset
    if opt.data =='facial' :
        #train_dataloader,test_dataloader = data_loader61(opt)
        train_dataloader, test_dataloader = data_loader3(opt, brain_region='temporal')  # 

    return train_dataloader,test_dataloader
