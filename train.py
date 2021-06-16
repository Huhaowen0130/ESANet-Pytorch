import os
import time
from datetime import datetime
import json
import sys
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.model import ESANet
from src.datasets.prepare_data import prepare_data
from src.func.args import parse_args
from src.func import utils
from src.func.utils import load_ckpt
from src.func.utils import save_ckpt, save_ckpt_every_epoch
from src.func.utils import ConfusionMatrixPytorch, miou_pytorch
from src.func.utils import print_log

def train():
    args = parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 模型路径
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, args.dataset,
                            f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, 'argsv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    
    # 训练数据、验证数据
    data_loaders = prepare_data(args, ckpt_dir)

    # if args.valid_full_res:
    #     train_loader, valid_loader, valid_loader_full_res = data_loaders
    # else:
    #     train_loader, valid_loader = data_loaders
    #     valid_loader_full_res = None
    train_loader, valid_loader = data_loaders

    cameras = train_loader.dataset.cameras
    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != 'None':
        class_weighting = train_loader.dataset.compute_class_weights(
            weight_mode=args.class_weighting,
            c=args.c_for_logarithmic_weighting)
    else:
        class_weighting = np.ones(n_classes_without_void)
    
    # log、mIoU（log只到mIoU）
    valid_split = valid_loader.dataset.split
    
    log_keys = [f'mIoU_{valid_split}']
    if args.valid_full_res:
        log_keys.append(f'mIoU_{valid_split}_full-res')
        # best_miou_full_res = 0
        
    log_keys_for_csv = log_keys.copy()
    
    for camera in cameras:
        log_keys_for_csv.append(f'mIoU_{valid_split}_{camera}')
        if args.valid_full_res:
            log_keys_for_csv.append(f'mIoU_{valid_split}_full-res_{camera}')
            
    confusion_matrices = dict()
    for camera in cameras:
        confusion_matrices[camera] = \
            ConfusionMatrixPytorch(n_classes_without_void)
        confusion_matrices['all'] = \
            ConfusionMatrixPytorch(n_classes_without_void)
    
    # 网络、loss、优化器
    net = ESANet.ESANet().to(device)
    print(net)
    
    # criterion = nn.CrossEntropyLoss(torch.tensor(class_weighting).to(device)).to(device)
    criterion = utils.CrossEntropyLoss2d(device, class_weighting)
    
    pixel_sum_valid_data = valid_loader.dataset.compute_class_weights(
        weight_mode='linear'
    )
    pixel_sum_valid_data_weighted = \
        np.sum(pixel_sum_valid_data * class_weighting)
    loss_function_valid = utils.CrossEntropyLoss2dForValidData(
        weight=class_weighting,
        weighted_pixel_sum=pixel_sum_valid_data_weighted,
        device=device
    )
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
    # checkpoint
    if args.last_ckpt:
        ckpt_path = os.path.join(ckpt_dir, args.last_ckpt)
        epoch_last_ckpt, best_miou, best_miou_epoch = \
            load_ckpt(net, optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0
    
    # 开始训练、验证
    for epoch in range(int(start_epoch), args.epochs):
        # 训练
        train_start = time.time()
        samples_of_epoch = 0
        
        net.train()
        for i, sample in enumerate(train_loader):
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = [sample['label'].to(device)]
            
            batch_size = image.data.shape[0]
            
            prediction = []
            prediction.append(net(depth, image))
            
            loss = sum(criterion(prediction, label))
            
            # optimizer.zero_grad()
            for param in net.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            
            samples_of_epoch += batch_size
    
            # print_log(epoch, samples_of_epoch, batch_size,
            #           len(train_loader.dataset), total_loss, time_inter,
            #           learning_rates)
            print_log(epoch, samples_of_epoch, batch_size,
                      len(train_loader.dataset), loss)
            
        train_time = time.time() - train_start
        print('batch_train: {:.0f}ms'.format(train_time * 1000))
            
        # 验证
        valid_start = time.time()
        
        miou = dict()
        
        loss_function_valid.reset_loss()
        
        net.eval()
        for camera in cameras:
            with valid_loader.dataset.filter_camera(camera):
                # confusion_matrices[camera].reset_conf_matrix()
                confusion_matrices[camera].reset()
                miou_temp = miou_pytorch(confusion_matrices[camera])
                print(f'{camera}: {len(valid_loader.dataset)} samples')
                
                for i, sample in enumerate(valid_loader):
                    depth = sample['depth'].to(device)
                    image = sample['image'].to(device)
                    prediction = net(depth, image)
                    
                    with torch.no_grad():
                        
                        # 归一化尺寸后的loss
                        loss_function_valid.add_loss_of_batch(
                            prediction,
                            sample['label'].to(device)
                        )
                        
                        # 用原始尺寸的图片计算验证集mIoU
                        label = sample['label_orig']
                        _, image_h, image_w = label.shape
                        
                        prediction = F.interpolate(
                            prediction,
                            (image_h, image_w),
                            mode='bilinear',
                            align_corners=False)
                        prediction = torch.argmax(prediction, dim=1)
                        
                        mask = label > 0
                        label = torch.masked_select(label, mask)
                        prediction = torch.masked_select(prediction,
                                                         mask.to(device))
                        
                        label -= 1
                        
                        prediction = prediction.cpu()
                        
                        # confusion_matrices[camera].update_conf_matrix(label,
                        #                                               prediction)
                        confusion_matrices[camera].update(label,
                                                          prediction, args.batch_size_valid)
                        
                # miou[camera], ious[camera] = \
                #     confusion_matrices[camera].compute_miou()
                miou[camera] = miou_temp.compute().data.numpy()
                print(f'mIoU {valid_split} {camera}: {miou[camera]}')
                
        confusion_matrices['all'].reset()
        miou_all_temp = miou_pytorch(confusion_matrices['all'])
        for camera in cameras:
            # confusion_matrices['all'].overall_confusion_matrix += \
            #     confusion_matrices[camera].overall_confusion_matrix
            confusion_matrices['all'].confusion_matrix += \
                confusion_matrices[camera].confusion_matrix
            confusion_matrices['all']._num_examples += confusion_matrices[camera]._num_examples
                
        # miou['all'] = confusion_matrices['all'].compute_miou()
        miou['all'] = miou_all_temp.compute().data.numpy()
        print(f"mIoU {valid_split}: {miou['all']}")
                            
        valid_time = time.time() - valid_start
        print('batch_valid: {:.0f}ms'.format(valid_time * 1000))
        
        # 保存模型
        print(miou['all'])
        save_current_checkpoint = False
        if miou['all'] > best_miou:
            best_miou = miou['all']
            best_miou_epoch = epoch
            save_current_checkpoint = True
            
        if epoch >= 10 and save_current_checkpoint is True:
            save_ckpt(ckpt_dir, net, optimizer, epoch)
            
        save_ckpt_every_epoch(ckpt_dir, net, optimizer, epoch, best_miou,
                              best_miou_epoch)
        
    print("Training completed ")

if __name__ == '__main__':
    train()