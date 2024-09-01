# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


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

class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        """
        初始化 PartNormalDataset 数据集类。

        参数:
        root (str): 数据集的根目录路径，默认值为 './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'。
        npoints (int): 每个点云样本的点数，默认值为 2500。
        split (str): 数据集的划分类型，可以是 'train'、'test' 或 'val'，默认值为 'train'。
        class_choice (list): 要选择的类别列表，如果为 None，则加载所有类别，默认值为 None。
        normal_channel (bool): 是否包含法向量信息，如果为 True，点云数据将包含法向量（6 维），否则仅包含坐标（3 维），默认值为 False。

        初始化步骤:
        1. 保存输入参数为类的属性。
        2. 构建类别映射文件的路径，并初始化类别字典。
        3. 根据是否包含法向量信息，决定点云数据的维度。

        属性:
        self.npoints (int): 每个点云样本的点数。
        self.root (str): 数据集的根目录路径。
        self.catfile (str): 类别映射文件 'synsetoffset2category.txt' 的路径。
        self.cat (dict): 存储类别名称到类别目录的映射字典。
        self.normal_channel (bool): 是否包含法向量信息。
        """
        self.npoints = npoints  # 每个点云样本的点数
        self.root = root  # 数据集的根目录路径
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # 类别映射文件路径
        self.cat = {}  # 初始化类别字典
        self.normal_channel = normal_channel  # 是否包含法向量信息


        # 打开并读取类别映射文件，将类别名称与对应的目录映射关系存储到 self.cat 字典中
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()  # 去除每行的首尾空白字符，并按空格分割成列表
                self.cat[ls[0]] = ls[1]  # 将类别名称映射到对应的类别目录

        # 创建一个类别名称到索引的映射字典 self.classes_original，索引用于标识每个类别
        self.cat = {k: v for k, v in self.cat.items()}  # 再次构建字典，确保类别映射是有序的（Python 3.7+ 字典默认有序）
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))  # 创建类别名称到索引的映射

        # 如果指定了 class_choice，则只保留 self.cat 中与 class_choice 匹配的类别
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)  # 可选的调试输出，打印类别映射关系


        # 初始化 self.meta 字典，用于存储每个类别的文件列表
        self.meta = {}

        # 读取并解析训练集文件列表，将文件 ID 存储到 train_ids 集合中
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        # 读取并解析验证集文件列表，将文件 ID 存储到 val_ids 集合中
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        # 读取并解析测试集文件列表，将文件 ID 存储到 test_ids 集合中
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        # 遍历每个类别
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []  # 初始化当前类别的文件列表
            dir_point = os.path.join(self.root, self.cat[item])  # 构建当前类别的文件夹路径
            fns = sorted(os.listdir(dir_point))  # 列出并排序当前类别文件夹中的所有文件名

            # 根据不同的分割（训练集、验证集、测试集）过滤文件名
            if split == 'trainval':
                # 如果是训练集或验证集，保留在 train_ids 或 val_ids 集合中的文件
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                # 如果是训练集，只保留在 train_ids 集合中的文件
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                # 如果是验证集，只保留在 val_ids 集合中的文件
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                # 如果是测试集，只保留在 test_ids 集合中的文件
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                # 如果分割类型未知，输出错误信息并退出程序
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # 遍历过滤后的文件名列表
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])  # 获取文件名（不包括扩展名）
                # 将文件的完整路径（带有 '.txt' 扩展名）添加到当前类别的文件列表中
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        # 初始化数据路径列表
        self.datapath = []
        # 遍历每个类别
        for item in self.cat:
            # 将类别名和对应的文件路径元组添加到数据路径列表中
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        # 初始化类别字典
        self.classes = {}
        # 将原始类别映射复制到当前类别字典中
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # 从类别名（如'Chair'）映射到一个整数列表，这些整数表示分割标签
        self.seg_classes = {
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

        # 输出每个类别及其对应的分割标签列表
        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        # 初始化缓存字典，用于将索引映射到 (point_set, cls, seg) 元组
        self.cache = {}
        # 设置缓存大小为20000
        self.cache_size = 20000


    def __getitem__(self, index):
        """
        根据索引获取点云数据及其对应的分类标签和分割标签。

        参数:
        index (int): 数据集的索引值。

        返回:
        tuple: 返回点云数据（point_set）、分类标签（cls）和分割标签（seg）。

        处理步骤:
        1. 检查索引对应的数据是否在缓存中。
            - 如果在缓存中，直接从缓存中获取点云数据、分类标签和分割标签。
            - 如果不在缓存中，加载数据文件。
        2. 根据 `normal_channel` 标志决定提取点云数据的哪些维度。
            - 如果 `normal_channel` 为 False，只提取前 3 维（x, y, z）。
            - 如果 `normal_channel` 为 True，提取前 6 维（x, y, z, nx, ny, nz）。
        3. 对点云数据进行归一化处理，使其中心化并缩放到单位球体内。
        4. 随机采样 npoints 个点，以确保输出的点云具有固定大小。
        5. 返回点云数据（point_set）、分类标签（cls）和分割标签（seg）。
        """
        if index in self.cache:
            point_set, cls, seg = self.cache[index]  # 从缓存中获取数据
        else:
            fn = self.datapath[index]  # 获取当前索引对应的数据文件路径
            cat = self.datapath[index][0]  # 获取分类类别
            cls = self.classes[cat]  # 获取分类标签
            cls = np.array([cls]).astype(np.int32)  # 将分类标签转换为整数数组
            data = np.loadtxt(fn[1]).astype(np.float32)  # 加载数据文件
            if not self.normal_channel:
                point_set = data[:, 0:3]  # 提取点云的前 3 维 (x, y, z)
            else:
                point_set = data[:, 0:6]  # 提取点云的前 6 维 (x, y, z, nx, ny, nz)
            seg = data[:, -1].astype(np.int32)  # 提取分割标签
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)  # 将数据存入缓存中

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])     # 对点云进行归一化处理

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # 随机重采样 npoints 个点
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        """
        获取数据集的总长度（样本数量）。

        返回:
        int: 数据集的总样本数量。
        """
        return len(self.datapath)



