

import os  
from PIL import Image
import numpy as np  

def rename_and_pair_images(rgb_folder, output_folder):  
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在
    
    # 按文件名排序以确保顺序一致（特别是timestamp命名时）
    image_files = sorted(os.listdir(rgb_folder))  

    for frame_index, filename in enumerate(image_files):
        file_path = os.path.join(rgb_folder, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 忽略非图像文件

        # 打开图像
        rgb_image = Image.open(file_path) 

        # 构造目标文件路径
        rgb_output_path = os.path.join(output_folder, f'frame-{frame_index:05d}.color.png')  

        # 保存重命名后的图像
        rgb_image.save(rgb_output_path)  
        print(f'Saved {rgb_output_path}')
    

  
def quaternion_to_rotation_matrix(q):  
    """  
    Convert quaternion coefficients (in w, x, y, z order) to a rotation matrix.  
    """  
    w, x, y, z = q  
    Nq = np.dot(q, q)  
    s = 2.0 / Nq  
    X = x * s  
    Y = y * s  
    Z = z * s  
    wX = w * X  
    wY = w * Y  
    wZ = w * Z  
    xX = x * X  
    xY = x * Y  
    xZ = x * Z  
    yY = y * Y  
    yZ = y * Z  
    zZ = z * Z  
  
    rotation_matrix = np.array([  
        [1 - (yY + zZ), xY - wZ, xZ + wY],  
        [xY + wZ, 1 - (xX + zZ), yZ - wX],  
        [xZ - wY, yZ + wX, 1 - (xX + yY)]  
    ], dtype=np.float64)  
  
    return rotation_matrix  
  
def find_closest_timestamp(timestamps, target_timestamp):  
    min_diff = float('inf')  
    closest_idx = None  
    for i, timestamp in enumerate(timestamps):  
        diff = abs(timestamp - target_timestamp)  
        if diff < min_diff:  
            min_diff = diff  
            closest_idx = i  
    return closest_idx  
  
def read_groundtruth(filename0, filename1):  
    poses = []  
    timestamp0, timestamp1 = [], []  
  
    # 读取两个文件的时间戳  
    with open(filename0, 'r') as file0:  
        next(file0)  # 跳过头部  
        for line in file0:  
            if not line.startswith("#"):  
                data = line.split()  
                timestamp0.append(float(data[0]))  
  
    with open(filename1, 'r') as file1:  
        next(file1)  # 跳过头部  
        for line in file1:  
            if not line.startswith("#"):  
                data = line.split()  
                timestamp1.append(float(data[0]))  
  
    # 对于filename1中的每个时间戳，找到filename0中最接近的pose  
    with open(filename0, 'r') as file0:  
        next(file0)  # 跳过头部  
        next(file0)
        next(file0)
        lines0 = file0.readlines()  # 读取所有行到列表中，以便通过索引访问  
  
    for i, t1 in enumerate(timestamp1):  
        closest_idx = find_closest_timestamp(timestamp0, t1)  
        if closest_idx is not None:  
            line = lines0[closest_idx]  
            data = line.split()  
            tx, ty, tz = float(data[1]), float(data[2]), float(data[3])  
            qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])  
            q = np.array([qw, qx, qy, qz])  
            t = np.array([tx, ty, tz, 1.0])  
            R = quaternion_to_rotation_matrix(q)  
            T = np.eye(4)  
            T[:3, :3] = R  
            T[:3, 3] = t[:3]  
            poses.append(T)  
  
    return poses  


rgb_folder = '/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/rgb'  # RGB图片文件夹路径  
#depth_folder = '/root/autodl-tmp/rgbd_dataset_freiburg1_desk2/rgbd_dataset_freiburg1_desk2/depth'  # 存储depth图片顺序的文件路径  
output_folder = '/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/order'  # 输出文件夹路径  
rename_and_pair_images(rgb_folder, output_folder)  

grouthtruth_txt='/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/groundtruth.txt'
rgb_txt="/mnt/HYX/rgbd_dataset_freiburg3_walking_rpy/rgb.txt"
poses = read_groundtruth(grouthtruth_txt,rgb_txt)  
for i, (poses) in enumerate(poses):  
    frame_number = f"{i:05d}"  # 将索引转换为5位数  
    filename = os.path.join(output_folder, f"frame-{frame_number}.pose.txt")     
    # 将变换矩阵写入文件  
    with open(filename, 'w') as file: 
        for row in poses:

            for i in row:
                file.write(f"{i:.7e}\t") 
            file.write("\n")          


