# 


## 训练
```bash
(py312forPointNet2) lee@192 pointnet2 % python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal --use_cpu
PARAMETER ...
Namespace(use_cpu=True, gpu='0', batch_size=30, model='pointnet2_cls_ssg', num_category=40, epoch=2, learning_rate=0.001, num_point=1024, optimizer='Adam', log_dir='pointnet2_cls_ssg_normal', decay_rate=0.0001, use_normals=True, process_data=False, use_uniform_sample=False)
Load dataset ...
The size of train data is 9843
The size of test data is 2468
No existing model, starting training from scratch...
Epoch 1 (1/2):
Train Instance Accuracy: 0.548171
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [05:09<00:00,  3.72s/it]
Saving at log/classification/pointnet2_cls_ssg_normal/checkpoints/best_model.pth
Test Instance Accuracy: 0.653614, Class Accuracy: 0.537815
Best Instance Accuracy: 0.653614, Class Accuracy: 0.000000
Epoch 2 (2/2):
Train Instance Accuracy: 0.704370
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [04:44<00:00,  3.43s/it]
Saving at log/classification/pointnet2_cls_ssg_normal/checkpoints/best_model.pth
Test Instance Accuracy: 0.769980, Class Accuracy: 0.679047
Best Instance Accuracy: 0.769980, Class Accuracy: 0.000000
```

## 测试
```bash
(py312forPointNet2) lee@192 pointnet2 % python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal --use_cpu
PARAMETER ...
Namespace(use_cpu=True, gpu='0', batch_size=24, num_category=40, num_point=1024, log_dir='pointnet2_cls_ssg_normal', use_normals=True, use_uniform_sample=False, num_votes=3)
Load dataset ...
The size of test data is 2468
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [11:32<00:00,  6.73s/it]
Test Instance Accuracy: 0.776618, Class Accuracy: 0.679727
```

## 分割训练
```bash
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
```

