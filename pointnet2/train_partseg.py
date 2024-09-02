"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset


import yaml  
  
# 读取YAML文件  
with open('config.yaml', 'r', encoding='utf-8') as file:  
    config = yaml.safe_load(file)  
  
# 提取字段  
ROOT_DIR = config['project_base_dir']  
ShapeNet_path = config['dataset']['ShapeNet_path'] 

 
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {
    # 'Earphone': [16, 17, 18], 
    # 'Motorbike': [30, 31, 32, 33, 34, 35], 
    # 'Rocket': [41, 42, 43],
    # 'Car': [8, 9, 10, 11], 
    # 'Laptop': [28, 29], 
    # 'Cap': [6, 7], 
    # 'Skateboard': [44, 45, 46], 
    # 'Mug': [36, 37],
    # 'Guitar': [19, 20, 21], 
    # 'Bag': [4, 5], 
    # 'Lamp': [24, 25, 26, 27], 
    # 'Table': [47, 48, 49],
    # 'Airplane': [0, 1, 2, 3],
    # 'Pistol': [38, 39, 40], 
    # 'Chair': [12, 13, 14, 15], 
    # 'Knife': [22, 23]
    'Workpiece': [0, 1, 2]
    }
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    # print(f"y: {y}")
    # print(f"num_classes: {num_classes}")
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch Size during training')
    parser.add_argument('--epoch', default=100, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')     # default=0.001
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''
    创建相关目录
    '''
    # 获取当前时间的字符串表示形式, 格式为 '年-月-日_时-分'
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    # 创建一个存放日志文件的目录 './log/'
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)

    # 在 './log/' 目录下创建一个名为 'part_seg' 的子目录, 用于存放分割任务的日志
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)

    # 如果用户没有指定日志目录, 则使用时间戳作为日志目录名称
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        # 如果用户指定了日志目录, 则使用用户指定的目录名
        exp_dir = exp_dir.joinpath(args.log_dir)

    # 创建日志目录（时间戳或用户指定的目录）
    exp_dir.mkdir(exist_ok=True)

    # 在日志目录下创建一个 'checkpoints/' 子目录, 用于保存模型的检查点
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    # 在日志目录下创建一个 'logs/' 子目录, 用于保存训练和测试的日志文件
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''
    日志设置
    '''
    args = parse_args()
    # 创建一个名为 "Model" 的日志记录器
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)   # 设置日志记录器的级别为 INFO
    # 定义日志格式, 包括时间、日志器名称、日志级别和消息内容
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建文件处理器, 用于将日志记录到指定的文件中
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)     # 设置文件处理器的日志级别为 INFO
    file_handler.setFormatter(formatter)    # 将格式应用到文件处理器
    logger.addHandler(file_handler)         # 将文件处理器添加到日志记录器

    # 记录一些基本信息
    log_string('PARAMETER ...')    
    # 记录命令行参数 
    log_string(args)

    # 指定数据集的根目录
    root = ShapeNet_path

    # 加载训练数据集
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # 记录训练集和测试集的数据量
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # 设置分类的类别数和分割的部分数
    num_classes = 16    # 数据集中类别的数量
    num_part = 7       # 每个对象可能的分割部分数

    '''
    模型加载
    '''
    # 动态导入指定的模型模块
    MODEL = importlib.import_module(args.model)
    # 复制模型代码文件到实验目录, 方便后续查看和重现实验
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    # 设置CUDA可见的设备（指定使用的GPU）, 根据命令行参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 检查是否有可用的CUDA设备, 并设置计算设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != '-1' else "cpu")
    # 如果CUDA可用, 打印当前使用的CUDA设备信息
    if torch.cuda.is_available():
        print(f"Current CUDA Device Index: {torch.cuda.current_device()}")  # 输出当前使用的CUDA设备索引
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")         # 输出第一个CUDA设备的名称
        print(f"Let's use { torch.cuda.device_count()} GPUs!")              # 输出CUDA设备的数量

    # 使用导入的模型定义分类器, 并将其移动到计算设备（GPU或CPU）
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).to(device)
    # 使用导入的模型定义损失函数, 并将其移动到计算设备
    criterion = MODEL.get_loss().to(device)
    # 为分类器的所有层应用inplace ReLU激活函数
    classifier.apply(inplace_relu)

    # 定义一个用于初始化权重的函数
    def weights_init(m):
        classname = m.__class__.__name__
        # 如果层的类名中包含 'Conv2d', 则使用 Xavier 正态分布初始化卷积层的权重, 并将偏置设置为 0
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        # 如果层的类名中包含 'Linear', 则使用 Xavier 正态分布初始化全连接层的权重, 并将偏置设置为 0
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    # 尝试加载预训练模型
    try:
        # 加载预训练模型的检查点
        pre_model_path = os.path.join(str(exp_dir) + '/checkpoints/best_model.pth')
        # pre_model_path = '/Users/lee/GitProjects/fork-pointnet2-pytorch/pointnet2/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth'
        # print('Use pretrain model: ' + pre_model_path)
        checkpoint = torch.load(pre_model_path)
        start_epoch = checkpoint['epoch']                               # 加载预训练模型的检查点
        classifier.load_state_dict(checkpoint['model_state_dict'])      # 加载模型的权重和偏置
        log_string('Use pretrain model')                                # 记录日志, 表示使用了预训练模型
    except:
        # 如果没有找到预训练模型，从头开始训练
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0                                 # 从头开始训练    
        classifier = classifier.apply(weights_init)     # 对模型应用自定义的权重初始化函数

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            # points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
                # points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)


