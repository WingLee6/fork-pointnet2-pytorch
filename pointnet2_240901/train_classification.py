"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import yaml  
  
# 读取YAML文件  
with open('config.yaml', 'r') as file:  
    config = yaml.safe_load(file)  
  
# 提取字段  
ROOT_DIR = config['project_base_dir']  
ShapeNet_path = config['dataset']['ModelNet_path'] 


# 获取当前文件所在目录的路径
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''参数解析函数'''
    parser = argparse.ArgumentParser('training')  # 创建一个解析器对象
    parser.add_argument('--use_cpu', action='store_true', default=False, help='使用CPU模式')  # 是否使用CPU
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')  # GPU设备编号
    parser.add_argument('--batch_size', type=int, default=30, help='训练时的批量大小')  # 批量大小, 默认24
    parser.add_argument('--model', default='pointnet_cls', help='模型名称 [默认: pointnet_cls]')  # 模型名称
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='在ModelNet10/40上训练')  # 类别数量
    parser.add_argument('--epoch', default=2, type=int, help='训练的轮数')  # 训练轮数, 默认200
    parser.add_argument('--learning_rate', default=0.001, type=float, help='训练的学习率')  # 学习率
    parser.add_argument('--num_point', type=int, default=1024, help='点的数量')  # 输入点的数量
    parser.add_argument('--optimizer', type=str, default='Adam', help='训练的优化器')  # 优化器类型
    parser.add_argument('--log_dir', type=str, default=None, help='实验的日志目录')  # 日志存放目录
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='权重衰减率')  # 权重衰减率
    parser.add_argument('--use_normals', action='store_true', default=False, help='使用法向量')  # 是否使用法向量
    parser.add_argument('--process_data', action='store_true', default=False, help='离线保存数据')  # 是否离线处理数据
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='使用均匀采样')  # 是否使用均匀采样
    return parser.parse_args()

def inplace_relu(m):
    '''替换ReLU为inplace ReLU的函数'''
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader, num_class=40):
    '''测试模型性能的函数'''
    mean_correct = []
    class_acc = np.zeros((num_class, 3))  # 初始化类别准确率矩阵
    classifier = model.eval()  # 将模型设置为评估模式

    # 遍历测试数据集
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:  # 如果不使用CPU，将数据转移到GPU
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)  # 转置点云数据
        pred, _ = classifier(points)  # 模型预测
        pred_choice = pred.data.max(1)[1]  # 获取预测类别

        # 计算每个类别的准确率
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()  # 计算准确预测的数量
        mean_correct.append(correct.item() / float(points.size()[0]))

    # 计算平均类别准确率和实例准确率
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

def main(args):
    '''主函数，负责训练和测试模型'''
    def log_string(str):
        logger.info(str)
        print(str)

    '''超参数设置'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置GPU设备

    '''创建目录结构'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))  # 获取当前时间作为字符串
    exp_dir = Path('./log/')  # 日志根目录
    exp_dir.mkdir(exist_ok=True)  # 创建日志目录
    exp_dir = exp_dir.joinpath('classification')  # 分类任务的日志目录
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)  # 如果未指定日志目录名，使用当前时间命名
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)  # 使用指定的日志目录名
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')  # 检查点保存目录
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')  # 日志文件保存目录
    log_dir.mkdir(exist_ok=True)

    '''日志配置'''
    args = parse_args()  # 解析命令行参数
    logger = logging.getLogger("Model")  # 创建日志记录器
    logger.setLevel(logging.INFO)  # 设置日志记录器的记录级别
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))  # 创建日志文件处理器
    file_handler.setLevel(logging.INFO)  # 设置日志文件处理器的记录级别
    file_handler.setFormatter(formatter)  # 应用日志格式到文件处理器
    logger.addHandler(file_handler)  # 将文件处理器添加到日志记录器
    log_string('PARAMETER ...')  # 记录参数信息
    log_string(args)

    '''数据加载'''
    log_string('Load dataset ...')
    data_path = ShapeNet_path

    # 加载训练和测试数据集
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''模型加载'''
    num_class = args.num_category  # 类别数量
    model = importlib.import_module(args.model)  # 动态导入模型模块
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))  # 复制模型代码到日志目录
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    # 获取模型和损失函数
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)  # 将ReLU替换为inplace ReLU

    if not args.use_cpu:  # 如果不使用CPU，将模型和损失函数转移到GPU
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        # 尝试加载预训练模型
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        # 如果没有预训练模型，则从头开始训练
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # 根据优化器选择进行优化器的设置
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # 设置学习率调度器，随着训练进行调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0  # 最好的实例准确率
    best_class_acc = 0.0  # 最好的类别准确率

    '''开始训练'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()  # 设置模型为训练模式

        '''训练循环'''
        for batch_id, (points, target) in enumerate(trainDataLoader):
            optimizer.zero_grad()  # 梯度清零

            if not args.use_cpu:  # 如果不使用CPU，将数据转移到GPU
                points, target = points.cuda(), target.cuda()

            points = points.data.numpy()  # 转换为numpy数组
            points = provider.random_point_dropout(points)  # 随机点丢弃
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])  # 随机缩放点云
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])  # 随机平移点云
            points = torch.Tensor(points)  # 转换为Tensor
            points = points.transpose(2, 1)  # 转置点云数据
            optimizer.zero_grad()  # 梯度清零
            pred, trans_feat = classifier(points)  # 模型预测
            loss = criterion(pred, target.long(), trans_feat)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新优化器

            correct = pred.argmax(dim=1).eq(target).sum().item()  # 计算准确预测的数量
            mean_correct.append(correct / float(points.size()[0]))
            global_step += 1

        # 计算每个epoch的实例准确率
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            # 测试模型在测试集上的表现
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
            # 更新保存最好的模型
            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
        scheduler.step()  # 调度学习率
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()  # 解析命令行参数
    main(args) 