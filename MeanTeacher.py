import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from Datasets.config import acdc_10_train_dataset, acdc_20_train_dataset
from Datasets.config import acdc_all_train_dataset, acdc_val_dataset
from Datasets.semi_datasets import TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

# 慢慢来，先好好理解理解
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=r"/home/baldwin/PLMT/Datasets/ACDC")
parser.add_argument("--exp", type=str, default="E1/ACDC/MeanTeacher", help="framework")
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--max_iterations", type=int, default=36000)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--patch_size", type=tuple, default=(256, 256))
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--labeled_bs", type=int, default=12)
parser.add_argument("--labeled_num", type=str, default="20")
parser.add_argument("--ema_decay", type=float, default=0.99)
parser.add_argument("--consistency", type=float, default=0.1)
parser.add_argument("--consistency_rampup", type=float, default=200.0)
args = parser.parse_args()

def patients_to_slices(dataset, patients_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"10":176, "20":342, "all":1692}
    else:
        print("Error")
    return ref_dict[str(patients_num)]

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_varialbes(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

# snapshot_path是存储信息所用的路径
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    ACDC_classes = {"1":"RV", "2":"MYO", "3":"LV"}

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    # 生成数据集确实不太好弄
    # 直接重新开辟一个文件，生成好了弄进来就行了
    if args.labeled_num == "10":
        db_train = acdc_10_train_dataset
    elif args.labeled_num == "20":
        db_train = acdc_20_train_dataset
    else:
        db_train = acdc_all_train_dataset
    db_val = acdc_val_dataset

    total_slice = len(acdc_all_train_dataset)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total slices is: {}, labeled slices is:{}".format(
        total_slice, labeled_slice))

    labeled_idx = list(range(0, labeled_slice))
    unlabeled_idx = list(range(labeled_slice, total_slice))
    batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, batch_size, batch_size-args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.01)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            if iter_num < 3000:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_varialbes(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info.loss_ce", loss_ce, iter_num)
            writer.add_scalar("info/loss_dice", loss_dice, iter_num)
            writer.add_scalar("info/consistency_loss", consistency_loss, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"],
                                                  sampled_batch["label"],
                                                  model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(ACDC_classes[str(class_i+1)]),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_jaccard'.format(ACDC_classes[str(class_i+1)]),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(ACDC_classes[str(class_i+1)]),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(ACDC_classes[str(class_i+1)]),
                                      metric_list[class_i, 3], iter_num)

                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_jaccard = np.mean(metric_list, axis=0)[1]
                mean_hd95 = np.mean(metric_list, axis=0)[2]
                mean_asd = np.mean(metric_list, axis=0)[3]
                writer.add_scalar("info/val_mean_dice", mean_dice, iter_num)
                writer.add_scalar("info/val_mean_jaccard", mean_jaccard, iter_num)
                writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)
                writer.add_scalar("info/val_mean_asd", mean_asd, iter_num)

                if mean_dice > best_performance:
                    best_performance = mean_dice
                    save_model_path = os.path.join(snapshot_path, "iter_{}_dice_{}.pth".
                                                   format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_model_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration: %d; mean_dice: %f; mean_jaccard: %f; mean_hd95: %f; mean_asd: %f'
                    % (iter_num, mean_dice, mean_jaccard, mean_hd95, mean_asd))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = r"/home/baldwin/PLMT/Experiments/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)






