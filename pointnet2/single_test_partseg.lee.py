import open3d as o3d
import numpy as np
import torch
import argparse
import importlib
import os
import sys

import yaml  
  
# 读取YAML文件  
with open('config.yaml', 'r', encoding='utf-8') as file:  
    config = yaml.safe_load(file)  
  
# 提取字段  
ROOT_DIR = config['project_base_dir']  
ShapeNet_path = config['dataset']['ShapeNet_path'] 

sys.path.append(os.path.join(ROOT_DIR, 'models'))

def to_categorical(y, num_classes):
    """将标签转换为1-hot编码"""
    # 创建一个大小为 (num_classes, num_classes) 的单位矩阵，每一行对应一个类别的1-hot编码
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    
    # 如果输入张量 y 在 GPU 上，则将生成的1-hot编码张量也转移到 GPU 上
    if (y.is_cuda):
        return new_y.cuda()
    
    # 如果输入张量 y 在 CPU 上，则返回 CPU 上的1-hot编码张量
    return new_y

def parse_args():
    '''参数解析'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='测试时的批处理大小')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')
    parser.add_argument('--num_point', type=int, default=2048, help='点云中的点数')
    parser.add_argument('--log_dir', type=str, required=True, help='实验的根目录')
    parser.add_argument('--normal', action='store_true', default=False, help='是否使用法线信息')
    parser.add_argument('--num_votes', type=int, default=3, help='通过投票聚合分割分数的次数')
    parser.add_argument('--ply_file', type=str, required=True, help='待测试的PLY文件路径')
    return parser.parse_args()

def load_ply(ply_file, num_point):
    """读取PLY文件并将点云数据预处理成模型输入格式"""
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    
    # 如果点云的点数大于 num_point，随机抽取 num_point 个点；否则填充到 num_point
    if len(points) > num_point:
        indices = np.random.choice(len(points), num_point, replace=False)
        points = points[indices]
    elif len(points) < num_point:
        padding = np.zeros((num_point - len(points), 3))
        points = np.vstack((points, padding))
    
    # 归一化处理
    points -= np.mean(points, axis=0)
    points /= np.max(np.sqrt(np.sum(points**2, axis=1)))
    
    return torch.from_numpy(points).float()

import numpy as np
import torch

def load_txt(txt_file, num_point):
    """读取TXT文件并将点云数据预处理成模型输入格式"""
    # 读取TXT文件中的点云数据
    points = np.loadtxt(txt_file)
    
    # 如果点云的点数大于 num_point，随机抽取 num_point 个点；否则填充到 num_point
    if len(points) > num_point:
        indices = np.random.choice(len(points), num_point, replace=False)
        points = points[indices]
    elif len(points) < num_point:
        padding = np.zeros((num_point - len(points), 3))
        points = np.vstack((points, padding))
    
    # 归一化处理
    points -= np.mean(points, axis=0)
    points /= np.max(np.sqrt(np.sum(points**2, axis=1)))
    
    return torch.from_numpy(points).float()

# 示例使用
# txt_file = 'path/to/your/pointcloud.txt'
# num_point = 2048
# points = load_txt(txt_file, num_point)

def main(args):
    # 设置CUDA可见的设备（指定使用的GPU）, 根据命令行参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != '-1' else "cpu")
    experiment_dir = 'log/part_seg/' + args.log_dir
    # 获取实验目录中 logs 文件夹下的模型名称（假设只有一个模型文件）
    
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print('Loading model: ' + model_name)
    MODEL = importlib.import_module(model_name)
    
    # 使用导入的模块创建分类器模型实例，传入参数为部件类别数量和是否使用法线信息
    num_classes = 16    # 数据集中有16个类别
    num_part = 50       # 总共有50个部件类别
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).to(device)
    
    # 加载保存的最佳模型检查点
    checkpoint = torch.load(experiment_dir + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # 读取PLY文件并预处理
    points = load_ply(args.ply_file, args.num_point)
    points = points.unsqueeze(0).to(device)  # 添加批次维度
    label = torch.zeros(1, args.num_point).long().to(device)  # 使用虚拟标签
    
    with torch.no_grad():
        # 切换模型到测试模式
        classifier = classifier.eval()  # 将模型设置为评估模式
        # 获取当前批次的数据维度
        cur_batch_size, NUM_POINT, _ = points.size()

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
        
    


    # # 初始化投票池
    # vote_pool = torch.zeros(1, args.num_point, num_part).to(device)
    # for _ in range(args.num_votes):
    #     seg_pred, _ = classifier(ply_points, ply_label)
    #     vote_pool += seg_pred
    
    # # 计算最终的预测结果
    # seg_pred = vote_pool / args.num_votes
    # cur_pred_val = seg_pred.cpu().data.numpy().squeeze()
    # predicted_labels = np.argmax(cur_pred_val, axis=1)

    # # 输出预测结果
    # print('PLY file prediction:')
    # print(predicted_labels)

if __name__ == '__main__':
    args = parse_args()
    main(args)




# python single_test_partseg.lee.py --batch_size 1 --gpu -1 --num_point 2048 --log_dir /Users/lee/GitProjects/fork-pointnet2-pytorch/pointnet2/log/part_seg/pointnet2_part_seg_msg_usemac --ply_file /Users/lee/GitProjects/fork-pointnet2-pytorch/pointnet2/cut1.ply