import cv2
import argparse
import numpy as np
from glob import glob
import py_sod_metrics


def eval(args):
    score = {}

    MAE = py_sod_metrics.MAE()
    FM = py_sod_metrics.Fmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()

    pred_list = glob(f"{args.pred_dir}/*")
    gt_list = list(args.mask_dir.glob('*'+args.mask_ext))
    pred_list.sort()
    gt_list.sort()

    for i in range(len(pred_list)):
        pred = cv2.imread(pred_list[i], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        gt = cv2.imread(gt_list[i], cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        MAE.step(pred=pred, gt=gt)
        FM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)

    mae = MAE.get_results()["mae"]
    mf = FM.get_results()["mf"]
    sm = SM.get_results()["sm"]

    print("MAE: {:.4f}, maxF: {:.4f}, S: {:.4f}".format(mae, mf, sm))

    score = {
        'MAE':mae,
        'F':mf,
        'S':sm,
        'total':1-mae+mf+sm
    }

    return score

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pre_dir", type=str, default="./predicts/")
    p.add_argument("--dataset_path", type=str, default="/home/sky/Dataset/2505/")
    p.add_argument("--dataset_list", type=list, default=['ESDIs','CrackSeg9k','ZJU-Leaper'])
    args = p.parse_args()

    scores = eval(args)