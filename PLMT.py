# -*-ing:utf-8-*-
import argparse
import logging
import os
import random
import shutil
import sys

import functorch.dim
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from Datasets.config import acdc_only_10_train_dataset, acdc_only_20_train_dataset, acdc_all_train_dataset
from Datasets.config import acdc_10_pseudo_train_dataset, acdc_20_pseudo_train_dataset
from Datasets.config import acdc_unlabeled_dataset, acdc_val_dataset
from Datasets.semi_datasets import TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

# 慢慢来，先好好理解理解
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=r"/home/baldwin/PLMT/Datasets/ACDC")
parser.add_argument("--exp", type=str, default="ACDC/PLMT", help="framework")
parser.add_argument("--etd", type=str, default="ACDC/PLMT", help="framework")
parser.add_argument("--K", type=int, default=500, help="framework")
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--patch_size", type=tuple, default=(256, 256))
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--labeled_bs", type=int, default=12)
parser.add_argument("--labeled_num", type=str, default="10")
parser.add_argument("--ema_decay", type=float, default=0.99)
parser.add_argument("--consistency", type=float, default=0.03)
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


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

# snapshot_path是存储信息所用的路径
# def train(args, snapshot_path):
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size
#     max_iterations = args.max_iterations
#
#     model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
#
#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)
#     # 生成数据集确实不太好弄
#     # 直接重新开辟一个文件，生成好了弄进来就行了
#     if args.labeled_num == "10":
#         db_train = la2018_only_10_train_dataset
#     elif args.labeled_num == "20":
#         db_train = la2018_only_20_train_dataset
#     else:
#         db_train = la2018_all_train_dataset
#     db_val = la2018_val_dataset
#
#     total_slice = len(db_train)
#     labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
#     print("Total slices is: {}, labeled slices is:{}".format(
#         total_slice, labeled_slice))
#
#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
#                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
#     valloader = DataLoader(db_val, batch_size=1, shuffle=True, num_workers=1)
#     model.train()
#
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)
#     ce_loss = CrossEntropyLoss()
#     dice_loss = losses.DiceLoss(num_classes)
#
#     writer = SummaryWriter(snapshot_path + "/log")
#     logging.info("{} iterations per epoch".format(len(trainloader)))
#
#     iter_num = 0
#     max_epoch = max_iterations // len(trainloader) + 1
#     best_performance = 0.0
#
#     iterator = tqdm(range(max_epoch), ncols=70)
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             volume_batch, label_batch = sampled_batch["image"], sampled_batch['label']
#             volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
#
#             outputs = model(volume_batch)
#             outputs_soft = torch.softmax(outputs, dim=1)
#
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
#
#             loss = 0.5 * (loss_dice + loss_ce)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = lr_
#
#             iter_num = iter_num + 1
#             writer.add_scalar("info/lr", lr_, iter_num)
#             writer.add_scalar("info/total_loss", loss, iter_num)
#             writer.add_scalar("info.loss_ce", loss_ce, iter_num)
#             writer.add_scalar("info/loss_dice", loss_dice, iter_num)
#
#             logging.info(
#                 'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
#                 (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
#
#             if iter_num > 0 and iter_num % 200 == 0:
#                 model.eval()
#                 metric_list = 0.0
#                 for i_batch, sampled_batch in enumerate(valloader):
#                     metric_i = test_single_volume(sampled_batch["image"],
#                                                   sampled_batch["label"],
#                                                   model, classes=num_classes)
#                     metric_list += np.array(metric_i)
#                 metric_list = metric_list / len(db_val)
#                 for class_i in range(num_classes - 1):
#                     writer.add_scalar('info/val_{}_dice'.format('LA2018'), metric_list[class_i, 0], iter_num)
#                     writer.add_scalar('info/val_{}_jaccard'.format('LA2018'), metric_list[class_i, 1], iter_num)
#                     writer.add_scalar('info/val_{}_hd95'.format('LA2018'), metric_list[class_i, 2], iter_num)
#                     writer.add_scalar('info/val_{}_asd'.format('LA2018'), metric_list[class_i, 3], iter_num)
#
#                 mean_dice = np.mean(metric_list, axis=0)[0]
#                 mean_jaccard = np.mean(metric_list, axis=0)[1]
#                 mean_hd95 = np.mean(metric_list, axis=0)[2]
#                 mean_asd = np.mean(metric_list, axis=0)[3]
#                 writer.add_scalar("info/val_mean_dice", mean_dice, iter_num)
#                 writer.add_scalar("info/val_mean_jaccard", mean_jaccard, iter_num)
#                 writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)
#                 writer.add_scalar("info/val_mean_asd", mean_asd, iter_num)
#
#                 if mean_dice > best_performance:
#                     best_performance = mean_dice
#                     save_model_path = os.path.join(snapshot_path, "iter_{}_dice_{}.pth".
#                                                    format(iter_num, round(best_performance, 4)))
#                     save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
#                     torch.save(model.state_dict(), save_model_path)
#                     torch.save(model.state_dict(), save_best)
#
#                 logging.info(
#                     'iteration: %d; mean_dice: %f; mean_jaccard: %f; mean_hd95: %f; mean_asd: %f'
#                     % (iter_num, mean_dice, mean_jaccard, mean_hd95, mean_asd))
#                 model.train()
#
#             if iter_num >= max_iterations:
#                 break
#         if iter_num >= max_iterations:
#             iterator.close()
#             break
#     writer.close()
#     return save_best


def label(model, best_model_path, dataset_path):
    model.load_state_dict(torch.load(best_model_path))
    print("init weight from {}".format(best_model_path))
    model.eval()
    train_model = acdc_unlabeled_dataset
    train_dataloader = DataLoader(train_model, batch_size=1, shuffle=True, num_workers=1)
    save_path = os.path.join(dataset_path, "pseudo")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i_batch, sampled_batch in enumerate(train_dataloader):
        volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

        outputs = model(volume_batch)
        outputs_soft = torch.softmax(outputs, dim=1)
        weight = outputs_soft.max(1)[0].cpu()
        mask = torch.BoolTensor((weight >= 0.9))
        pseudo_label = torch.argmax(outputs_soft, dim=1).cpu()
        pseudo_label = pseudo_label.numpy().astype(np.uint8)
        with h5py.File(os.path.join(save_path, sampled_batch["id"][0]), "w") as hf:
            hf.create_dataset("pseudo", data=pseudo_label[0, ...])
            hf.create_dataset("mask", data=mask[0, ...])

    return "Finish pseudo label creation"

# # 这个地方就是相当于重新训练一下
def second_train(args, snapshot_path):
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
    ema_model = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.labeled_num == "10":
        db_train = acdc_10_pseudo_train_dataset
    elif args.labeled_num == "20":
        db_train = acdc_20_pseudo_train_dataset
    else:
        db_train = acdc_all_train_dataset
    db_val = acdc_val_dataset

    total_slice = len(acdc_all_train_dataset)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total slices is: {}, labeled slices is: {}".format(
        total_slice, labeled_slice))

    labeled_idx = list(range(0, labeled_slice))
    unlabeled_idx = list(range(labeled_slice, total_slice))
    batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, batch_size, batch_size-args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=True, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.005)
    log_weights = nn.Parameter(torch.randn(2))
    optimizer.add_param_group({"params": log_weights})

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_Batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, mask_batch = sampled_batch["image"], sampled_batch["label"], sampled_batch["mask"]
            volume_batch, label_batch, mask_batch = volume_batch.cuda(), label_batch.cuda(), mask_batch.cuda()
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
            consistency_weight = get_current_consistency_weight(iter_num // 200)
            weights = F.softmax(log_weights, dim=0)

            if iter_num < 5000:
                unsupervised_loss = 0.0
                pseudo_sup_loss = 0.0
                consistency_loss = 0.0
            else:
                pseudo_sup_loss = losses.confidence_ce_loss(outputs_soft[args.labeled_bs:],
                                                              label_batch[args.labeled_bs:].unsqueeze(1),
                                                              mask_batch[args.labeled_bs:])
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)

                unsupervised_loss = weights[0] * pseudo_sup_loss + weights[1] * args.K * consistency_loss


            loss = supervised_loss + consistency_weight * unsupervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)
            writer.add_scalar("info/loss_dice", loss_dice, iter_num)
            if iter_num < 5000:
                writer.add_scalar("info/weight_0", weights[0], iter_num)
                writer.add_scalar("info/weight_1", weights[1], iter_num)
                writer.add_scalar("info/pseudo_sup_loss", pseudo_sup_loss, iter_num)
                writer.add_scalar("info/consistency_loss", consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, '
                'weight_0: %f, weight_1: %f, pseudo_sup_loss: %f, consistency_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(),
                 weights[0], weights[1], pseudo_sup_loss, consistency_loss))

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




def run():
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

    snapshot_path = r"/home/baldwin/PLMT/Experiments/{}_{}_labeled_K_{}/{}".format(
        args.exp, args.labeled_num, str(args.K), args.model
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model_path = r"/home/baldwin/PLMT/Experiments/{}/FullSupervision_{}_labeled/{}".format(
        args.etd, 
        args.labeled_num
        args.model
    )
    bset_model_path = os.path.join(model_path, "unet_best_model.pth")
    print("First training stage Finished!")

    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)

    label(model, bset_model_path, args.root_path)
    print("Second Pseudo Label Creation Stage Finished")

    second_train(args, snapshot_path)
    save_path = os.path.join(args.root_path, "pseudo")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    print("SelfTraining Finished!")


if __name__ == "__main__":
    run()



