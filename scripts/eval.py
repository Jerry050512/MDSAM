import os
import sys
import cv2
import numpy as np
from glob import glob
import py_sod_metrics

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config import (
    TEST_MASK_DIR,
    MASK_EXT,
)

def evaluate(pred_dir):
    """
    Evaluate the predicted masks against the ground truth.

    Args:
        pred_dir (str or Path): Directory containing the predicted masks.
    """
    print(f"--- Evaluating predictions in {pred_dir} ---")

    # ------- 1. Metrics setup -------
    MAE = py_sod_metrics.MAE()
    FM = py_sod_metrics.Fmeasure()
    SM = py_sod_metrics.Smeasure()

    # ------- 2. Load predictions and ground truth -------
    pred_list = sorted(glob(f"{pred_dir}/*{MASK_EXT}"))
    gt_list = sorted(list(TEST_MASK_DIR.glob(f'*{MASK_EXT}')))

    if not pred_list:
        print("Warning: No predictions found in the specified directory.")
        return None
    if not gt_list:
        print("Warning: No ground truth masks found.")
        return None
    if len(pred_list) != len(gt_list):
        print(f"Warning: Number of predictions ({len(pred_list)}) and ground truth ({len(gt_list)}) masks do not match.")
        return None

    # ------- 3. Calculate metrics -------
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        MAE.step(pred=pred, gt=gt)
        FM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)

    mae = MAE.get_results()["mae"]
    mf = FM.get_results()["mf"]
    sm = SM.get_results()["sm"]

    print(f"MAE: {mae:.4f}, maxF: {mf:.4f}, S: {sm:.4f}")

    score = {
        'MAE': mae,
        'F': mf,
        'S': sm,
        'total': 1 - mae + mf + sm
    }

    print("--- Evaluation Done ---")
    return score

if __name__ == "__main__":
    # This is an example of how to run the evaluation script.
    # You would typically call this after running the test script.
    # For example:
    # from config import SAVE_ROOT, MODEL_NAME, TEST_DATASET
    # epoch_to_eval = 80
    # pred_dir = SAVE_ROOT / f"{MODEL_NAME}-epoch-{epoch_to_eval}" / f"{TEST_DATASET}"
    # evaluate(pred_dir)

    print("This script is intended to be called with a specific prediction directory.")
    print("Please modify the __main__ block to evaluate a set of predictions.")
