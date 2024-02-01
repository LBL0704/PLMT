# -*-ing:utf-8-*-
# -*-ing:utf-8-*-
import argparse
import os
import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from torch.utils.data import DataLoader
from Datasets.config import acdc_val_dataset

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r"/home/baldwin/PLMT/Datasets/ACDC", help='Name of Experiment')
parser.add_argument("--K", type=int, default=500, help="framework")
parser.add_argument('--exp', type=str,
                    default='E1/ACDC/MeanTeacher', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=str, default="10",
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = True
    gt[gt > 0] = True
    pred[pred == 0] = False
    gt[gt == 0] = False
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jaccard, hd95, asd
    else:
        return 0, 0, 0, 0

def test_single_volume(case, net, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/test/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model in ["unet_urpc", "unet_cct"]:
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    rv_metrics = calculate_metric_percase(prediction == 1, label == 1)
    myo_metrics = calculate_metric_percase(prediction == 2, label == 2)
    lv_metrics = calculate_metric_percase(prediction == 3, label == 3)
    return rv_metrics, myo_metrics, lv_metrics


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = r"/home/baldwin/PLMT/Experiments/{}_{}_labeled_K_{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, str(FLAGS.K), FLAGS.model
    )
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    rv_total = 0.0
    myo_total = 0.0
    lv_total = 0.0
    for case in tqdm(image_list):
        RV_metrics, MYO_metrics, LV_metrics = test_single_volume(case, net, FLAGS)
        rv_total += np.asarray(RV_metrics)
        myo_total += np.asarray(MYO_metrics)
        lv_total += np.asarray(LV_metrics)
    avg_metric = [rv_total / len(image_list), myo_total /
                  len(image_list), lv_total / len(image_list)]

    BestPerformanceFile = open(os.path.join(snapshot_path, r"BestPerformance.txt"), "a")

    BestPerformanceFile.write("RV dice: " + str(avg_metric[0][0])+"\n")
    BestPerformanceFile.write("RV jaccard: " + str(avg_metric[0][1])+"\n")
    BestPerformanceFile.write("RV HD95: " + str(avg_metric[0][2])+"\n")
    BestPerformanceFile.write("RV asd: " + str(avg_metric[0][3])+"\n")

    BestPerformanceFile.write("MYO dice: " + str(avg_metric[1][0])+"\n")
    BestPerformanceFile.write("MYO jaccard: " + str(avg_metric[1][1])+"\n")
    BestPerformanceFile.write("MYO HD95: " + str(avg_metric[1][2])+"\n")
    BestPerformanceFile.write("MYO asd: " + str(avg_metric[1][3])+"\n")

    BestPerformanceFile.write("LV dice: " + str(avg_metric[2][0])+"\n")
    BestPerformanceFile.write("LV jaccard: " + str(avg_metric[2][1])+"\n")
    BestPerformanceFile.write("LV HD95: " + str(avg_metric[2][2])+"\n")
    BestPerformanceFile.write("LV asd: " + str(avg_metric[2][3])+"\n")
    return "Test Finished"


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)

