"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pathlib import Path  
  
# 读取YAML文件  
with open('config.yaml', 'r', encoding='utf-8') as file:  
    config = yaml.safe_load(file)  
  
# 提取字段  
ROOT_DIR = config['project_base_dir']  
ShapeNet_path = config['dataset']['ShapeNet_path'] 

sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 定义一个字典，映射每个类别名称到其对应的标签列表
seg_classes = {
    'Earphone': [16, 17, 18], 
    'Motorbike': [30, 31, 32, 33, 34, 35], 
    'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11], 
    'Laptop': [28, 29], 
    'Cap': [6, 7], 
    'Skateboard': [44, 45, 46], 
    'Mug': [36, 37],
    'Guitar': [19, 20, 21], 
    'Bag': [4, 5], 
    'Lamp': [24, 25, 26, 27], 
    'Table': [47, 48, 49],
    'Airplane': [0, 1, 2, 3], 
    'Pistol': [38, 39, 40], 
    'Chair': [12, 13, 14, 15], 
    'Knife': [22, 23]
}

# 创建一个字典，将每个标签映射到对应的类别名称
seg_label_to_cat = {}  # 格式为 {标签: 类别名称}
for cat in seg_classes.keys():  # 遍历每个类别名称
    for label in seg_classes[cat]:  # 遍历每个类别下的标签
        seg_label_to_cat[label] = cat  # 将标签映射到类别名称
        

def pc_normalize(pc):
    """
    对点云数据进行归一化处理。

    参数:
    pc (numpy.ndarray): 输入的点云数据，形状为 (N, 3)，其中 N 是点的数量。

    返回:
    numpy.ndarray: 归一化后的点云数据，形状与输入相同。

    处理步骤:
    1. 计算点云数据的质心（所有点的平均值）。
    2. 将点云数据的每个点减去质心，使得点云中心化。
    3. 计算所有点到质心的欧氏距离的最大值。
    4. 将点云数据的每个点除以最大距离，将点云缩放到单位球体内。
    """
    centroid = np.mean(pc, axis=0)  # 计算点云的质心
    pc = pc - centroid  # 将点云中心化，使质心移动到原点
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 计算所有点到质心的最大距离
    pc = pc / m  # 将点云缩放到单位球体内
    return pc

def to_categorical(y, num_classes):
    """将标签转换为1-hot编码"""
    # print(f"y: {y}")
    # print(f"num_classes: {num_classes}")
    # 创建一个大小为 (num_classes, num_classes) 的单位矩阵，每一行对应一个类别的1-hot编码
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    
    # 如果输入张量 y 在 GPU 上，则将生成的1-hot编码张量也转移到 GPU 上
    if (y.is_cuda):
        return new_y.cuda()
    
    # 如果输入张量 y 在 CPU 上，则返回 CPU 上的1-hot编码张量
    return new_y


def generate_random_point_cloud(batch_size=1, num_points=2048, 
                                num_features=6, num_classes=40):
    """
    生成一组随机的点云数据、类别标签和分割标签。

    Parameters:
    - batch_size: 批量大小
    - num_points: 点云中的点数
    - num_features: 每个点的特征数
    - num_classes: 类别数量

    Returns:
    - points: 点云数据，形状为 [batch_size, num_points, num_features]
    - label: 类别标签，形状为 [batch_size]
    - target: 分割标签，形状为 [batch_size, num_points]
    """
    # 随机生成点云数据
    points = np.random.rand(batch_size, num_points, num_features).astype(np.float32)
    
    # 随机生成类别标签
    label = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    
    # 随机生成分割标签
    target = np.random.randint(0, 3, size=(batch_size, num_points)).astype(np.int32)
    
    # 转换为 PyTorch 张量
    points = torch.from_numpy(points)
    label = torch.from_numpy(label)
    target = torch.from_numpy(target)
    
    return points, label, target

def load_point_cloud_from_txt(file_path):
    """
    从一个 .txt 文件中读取点云数据，并将其转换为 PyTorch 张量。

    Parameters:
    - file_path: 点云数据文件的路径

    Returns:
    - points: 点云数据，形状为 [num_points, num_features]，其中 num_features = 6
    - normals: 法向量，形状为 [num_points, 3]
    - labels: 标签，形状为 [num_points]
    """
    # 读取文件内容
    data = np.loadtxt(file_path).astype(np.float32)  # 加载数据文件
    if not args.normal:  # 如果不使用法向量，则只提取点云的前 3 维 (x, y, z)
        point_set = data[:, 0:3]  # 提取点云的前 3 维 (x, y, z)
    else:
        point_set = data[:, 0:6]  # 提取点云的前 6 维 (x, y, z, nx, ny, nz)

    # 提取分割标签
    seg = data[:, -1].astype(np.int32)

    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])     # 对点云进行归一化处理
    
    choice = np.random.choice(len(seg), args.num_point, replace=True)  # 随机选择 args.num_point 个点
    # 随机重采样 npoints 个点
    point_set = point_set[choice, :]
    seg = seg[choice]
    
    cls = np.array([0]).astype(np.int32)  # 将分类标签转换为整数数组, 只是随便设个0

    return point_set, cls, seg



def plot_3d_points(points: torch.Tensor, labels: np.ndarray):
    """
    绘制带有标签的三维点云图。

    参数:
    - points: torch.Tensor, 形状为(N, 3)的三维坐标。
    - labels: np.ndarray, 形状为(N,)的标签数组。
    """
    # 确保points是numpy数组
    points_np = points.numpy()

    # 创建图形和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    scatter = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=labels, cmap='viridis')

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Labels')

    # 设置坐标轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # 显示图形
    plt.show()


def parse_args():
    '''
    参数解析
    '''
    parser = argparse.ArgumentParser('PointNet')  # 创建ArgumentParser对象，用于处理命令行参数
    parser.add_argument('--batch_size', type=int, default=24, help='测试时的批处理大小')  # 添加batch_size参数，默认为24
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')  # 添加gpu参数，用于指定使用的GPU设备
    parser.add_argument('--num_point', type=int, default=2048, help='点云中的点数')  # 添加num_point参数，默认为2048，用于指定点云中的点数
    parser.add_argument('--log_dir', type=str, required=True, help='实验的根目录')  # 添加log_dir参数，必须指定，用于指定实验日志的根目录
    parser.add_argument('--normal', action='store_true', default=False, help='是否使用法线信息')  # 添加normal参数，用于选择是否使用法线信息
    parser.add_argument('--num_votes', type=int, default=3, help='通过投票聚合分割分数的次数')  # 添加num_votes参数，默认为3，用于指定投票次数
    parser.add_argument('--point_cloud_file', type=str, default='random', help='点云数据集的路径')  # 添加point_cloud参数，默认为random(用随机数据)，用于指定点云数据集的路径
    return parser.parse_args()  # 解析并返回命令行参数


def main(args):
    '''HYPER PARAMETER'''
    # 设置CUDA可见的设备（指定使用的GPU）, 根据命令行参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 检查是否有可用的CUDA设备, 并设置计算设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != '-1' else "cpu")
    experiment_dir = 'log/part_seg/' + args.log_dir

    
    if args.point_cloud_file == 'random':
        # 随机生成点云数据
        points, label, target = generate_random_point_cloud(batch_size=1, 
                                                            num_points=2048, 
                                                            num_features=6, 
                                                            num_classes=4)
    else:
        file_path = Path(args.point_cloud_file)
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件 '{file_path}' 不存在。")
        
        if file_path.suffix.lower() in ['.txt']:
            # 从文件中读取点云数据
            points, label, target = load_point_cloud_from_txt(file_path)
            print("pointcloud shape: ", points.shape)
            print("label shape: ", label.shape)
            print("target shape: ", target.shape)
            points = torch.tensor(points)
            label = torch.tensor(label)
            target = torch.tensor(target)
            
            # 增加一个维度
            points = points.unsqueeze(0)  # 增加一个维度，使其形状为 [batch_size, num_points, num_features]
            label = label.unsqueeze(0)  # 增加一个维度，使其形状为 [batch_size]
            target = target.unsqueeze(0)  # 增加一个维度，使其形状为 [batch_size, num_points]

    print("pointcloud shape: ", points.shape)
    print("label shape: ", label.shape)
    print("target shape: ", target.shape)
    cur_batch_size, NUM_POINT, _ = points.size()
    # print('cur_batch_size:', cur_batch_size)
    # print('NUM_POINT:', NUM_POINT)
    # print('target:', target)
    # print('target:', target.size())
    # from collections import Counter
    # # 统计每个值出现的次数
    # target_unique_values = torch.unique(target)             # 不同的值
    # target_num_part = target_unique_values.numel()          # 不同值的数量

    # label_unique_values = torch.unique(label)               # 不同的值
    # label_num_unique_values = label_unique_values.numel()   # 不同值的数量

    
    num_classes = 16 # 数据集中有16个类别
    num_part = 50  # 总共有部件类别

    '''
    模型加载
    '''
    # 获取实验目录中 logs 文件夹下的模型名称（假设只有一个模型文件）
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print('Loading model: ' + model_name)
    # 动态导入模型模块
    MODEL = importlib.import_module(model_name)
    
    # 使用导入的模块创建分类器模型实例，传入参数为部件类别数量和是否使用法线信息
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).to(device)
    
    # 加载保存的最佳模型检查点
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    
    # 将检查点中的模型状态字典加载到当前模型实例中
    classifier.load_state_dict(checkpoint['model_state_dict'])
    

    with torch.no_grad():
        # 模型设置
        classifier = classifier.eval()  # 将模型设置为评估模式
        # 将点云数据、标签和目标转换为浮点型并移动到设备（如GPU）
        points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)

        # 转置点云数据的维度以匹配模型输入的要求
        points = points.transpose(2, 1)

        # 初始化投票池，用于存储多个投票轮次的分割预测
        vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).to(device)

        # 多轮投票预测以增强稳定性
        for _ in range(args.num_votes):
            seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            vote_pool += seg_pred  # 将每轮的预测累加到投票池中

        # 将投票池中的预测结果取平均，得到最终的预测结果
        seg_pred = vote_pool / args.num_votes

        # 将预测结果从 GPU 转移到 CPU，并转换为 NumPy 数组
        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val  # 保留未处理的逻辑回归结果

        # 初始化一个零矩阵用于存储最终的预测结果
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)

        # 将目标标签从 GPU 转移到 CPU，并转换为 NumPy 数组
        target = target.cpu().data.numpy()

        # 对每一个点云实例进行处理
        # for i in range(cur_batch_size):
        # 获取当前实例的类别
        cat = seg_label_to_cat[target[0, 0]]
        
        # 获取当前实例的预测逻辑回归结果
        logits = cur_pred_val_logits[0, :, :]
        print('logits:', logits.shape)

        # 对该实例进行预测，并根据类别的标签范围调整预测结果
        cur_pred_val[0, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
        # 打印预测的分割标签
        print(f"点云实例的分割标签:")
        print(cur_pred_val[0, :])
        print(len(cur_pred_val[0, :]))

        # 点云可视化
        plot_3d_points(points, cur_pred_val[0, :])


if __name__ == '__main__':
    args = parse_args()
    main(args)




# python single_test_partseg.lee.py --normal --log_dir pointnet2_part_seg_msg_usemac --gpu -1 --point_cloud_file /Volumes/data/Datasets/PointCloudsDatasets/ShapeNet_normal/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02954340/1fc6f3aebca014ab19ba010ddb4974fe.txt
# python single_test_partseg.lee.py --normal --log_dir pointnet2_part_seg_msg_usemac --gpu -1 --point_cloud_file /Volumes/data/Datasets/PointCloudsDatasets/welding_workpiece_l515/data/label8.txt

