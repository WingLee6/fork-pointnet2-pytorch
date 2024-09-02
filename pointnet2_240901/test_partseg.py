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

import yaml  
  
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
    parser = argparse.ArgumentParser('PointNet')  # 创建ArgumentParser对象，用于处理命令行参数
    parser.add_argument('--batch_size', type=int, default=24, help='测试时的批处理大小')  # 添加batch_size参数，默认为24
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')  # 添加gpu参数，用于指定使用的GPU设备
    parser.add_argument('--num_point', type=int, default=2048, help='点云中的点数')  # 添加num_point参数，默认为2048，用于指定点云中的点数
    parser.add_argument('--log_dir', type=str, required=True, help='实验的根目录')  # 添加log_dir参数，必须指定，用于指定实验日志的根目录
    parser.add_argument('--normal', action='store_true', default=False, help='是否使用法线信息')  # 添加normal参数，用于选择是否使用法线信息
    parser.add_argument('--num_votes', type=int, default=3, help='通过投票聚合分割分数的次数')  # 添加num_votes参数，默认为3，用于指定投票次数
    return parser.parse_args()  # 解析并返回命令行参数


def main(args):
    # 定义一个内部函数 log_string，用于记录日志信息并将信息打印到控制台
    def log_string(str):
        logger.info(str)  # 记录日志信息
        print(str)        # 将日志信息打印到控制台

    '''HYPER PARAMETER'''
    # 设置CUDA可见的设备（指定使用的GPU）, 根据命令行参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 检查是否有可用的CUDA设备, 并设置计算设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != '-1' else "cpu")
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''
    日志设置
    '''
    args = parse_args()
    logger = logging.getLogger("Model")  # 创建一个记录器，名称为 "Model"
    logger.setLevel(logging.INFO)  # 设置日志的最低级别为 INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 定义日志的格式，包括时间、名称、日志级别和消息内容
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)  # 创建一个文件处理器，将日志保存到指定路径的 eval.txt 文件中
    file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别为 INFO
    file_handler.setFormatter(formatter)  # 将格式应用到文件处理器上
    logger.addHandler(file_handler)  # 将文件处理器添加到记录器中
    log_string('PARAMETER ...')  # 记录一条信息，表明参数部分的日志开始
    log_string(args)  # 记录解析后的命令行参数

    root = ShapeNet_path  # 数据集的根目录路径，从配置文件中读取

    # 加载测试数据集
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, 
                                     split='test', normal_channel=args.normal)
    # 创建数据加载器，用于批量加载测试数据
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=False, num_workers=4)
    # 记录测试数据的数量
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    
    num_classes = 16  # 数据集中有16个类别
    num_part = 50  # 总共有50个部件类别

    '''MODEL LOADING'''
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
        # 在不计算梯度的上下文中执行，节省内存，适用于评估阶段
        test_metrics = {}  # 存储评估指标
        total_correct = 0  # 记录总体的正确预测数
        total_seen = 0  # 记录总的预测点数
        total_seen_class = [0 for _ in range(num_part)]  # 记录每个部件类别的总预测点数
        total_correct_class = [0 for _ in range(num_part)]  # 记录每个部件类别的正确预测点数
        shape_ious = {cat: [] for cat in seg_classes.keys()}  # 存储每个类别的形状IoU（交并比）
        seg_label_to_cat = {}  # 建立从分割标签到类别的映射

        # 构建分割标签到类别的映射字典
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()  # 将模型设置为评估模式

        # 遍历测试数据集，进行批量预测
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            print("pointcloud shape: ", points.shape)
            print("label shape: ", label.shape)
            print("target shape: ", target.shape)
            # 获取当前批次的数据维度
            batchsize, num_point, _ = points.size()
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

            # 将预测结果从 GPU 转移到 CPU，并转换为 NumPy 数组
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val  # 保留未处理的逻辑回归结果

            # 初始化一个零矩阵用于存储最终的预测结果
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)

            # 将目标标签从 GPU 转移到 CPU，并转换为 NumPy 数组
            target = target.cpu().data.numpy()

            # 对每一个点云实例进行处理
            for i in range(cur_batch_size):
                # 获取当前实例的类别
                cat = seg_label_to_cat[target[i, 0]]
                
                # 获取当前实例的预测逻辑回归结果
                logits = cur_pred_val_logits[i, :, :]

                # 对该实例进行预测，并根据类别的标签范围调整预测结果
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            # 计算当前批次中正确预测的点数
            correct = np.sum(cur_pred_val == target)
            total_correct += correct  # 累加正确预测的点数
            total_seen += (cur_batch_size * NUM_POINT)  # 累加总的点数

            # 计算每个部件类别的正确预测和总预测点数
            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)  # 累加该类别的总点数
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))  # 累加该类别的正确预测点数


            # 对当前批次中的每个点云实例进行处理
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]  # 当前实例的预测分割结果
                segl = target[i, :]  # 当前实例的真实标签
                cat = seg_label_to_cat[segl[0]]  # 获取当前实例的类别

                # 初始化一个列表，用于存储该实例中每个部分的IoU（交并比）
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

                # 计算每个部分的IoU
                for l in seg_classes[cat]:
                    # 如果该部分在真实标签和预测结果中都不存在，则IoU为1.0
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        # 否则，计算该部分的IoU
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))

                # 将当前实例的平均IoU加入到类别对应的IoU列表中
                shape_ious[cat].append(np.mean(part_ious))

        # 初始化一个列表，用于存储所有实例的IoU值
        all_shape_ious = []
        
        # 遍历每个类别，计算平均IoU
        for cat in shape_ious.keys():
            # 将当前类别的所有IoU值加入到总的IoU列表中
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            # 计算当前类别的平均IoU，并更新shape_ious字典
            shape_ious[cat] = np.mean(shape_ious[cat])
        
        # 计算所有类别的平均IoU
        mean_shape_ious = np.mean(list(shape_ious.values()))

        # 计算总体准确率
        test_metrics['accuracy'] = total_correct / float(total_seen)

        # 计算每个类别的平均准确率
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))

        # 打印并记录每个类别的平均IoU
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))

        # 将平均IoU记录到测试指标中
        test_metrics['class_avg_iou'] = mean_shape_ious

        # 计算并记录实例平均IoU
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    # 记录总体准确率，并打印精确到小数点后五位
    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])

    # 记录每个类别的平均准确率，并打印精确到小数点后五位
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])

    # 记录每个类别的平均IoU，并打印精确到小数点后五位
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])

    # 记录实例平均IoU，并打印精确到小数点后五位
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
