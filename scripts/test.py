import os
import sys
import torch
from tqdm import tqdm
import cv2
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset.sod_dataset import getDefectDataloader
from model import DefectSAM
from config import (
    DEVICE,
    MODEL_NAME,
    IMG_SIZE,
    SAVE_ROOT,
    TEST_IMG_DIR,
    TEST_MASK_DIR,
    TEST_DATASET,
    IMG_EXT,
    MASK_EXT,
)

def test(model_path, epoch=0):
    """
    Test the DefectSAM model on the test dataset.

    Args:
        model_path (str or Path): Path to the trained model checkpoint.
        epoch (int): The epoch number of the checkpoint, used for saving results.
    """
    # ------- 1. Model setup -------
    print("--- Setting up model ---")
    net = DefectSAM(IMG_SIZE)
    net.load_state_dict(torch.load(model_path)["model"])
    net.to(DEVICE)
    net.eval()

    # ------- 2. Dataloader setup -------
    print("--- Setting up dataloader ---")
    class DataloaderConfig:
        def __init__(self):
            self.mode = "test"
            self.img_dir = TEST_IMG_DIR
            self.mask_dir = TEST_MASK_DIR
            self.img_ext = IMG_EXT
            self.mask_ext = MASK_EXT
            self.batch_size = 1  # Test with batch size 1
            self.num_workers = 0

    dataloader_cfg = DataloaderConfig()
    dataloader_test = getDefectDataloader(dataloader_cfg, img_size=IMG_SIZE)

    # ------- 3. Output directory setup -------
    save_path = SAVE_ROOT / f"{MODEL_NAME}-epoch-{epoch}" / f"{TEST_DATASET}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"--- Saving predictions to {save_path} ---")

    # ------- 4. Start testing -------
    print("------- Start Testing -------")
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for data in tqdm(dataloader_test, desc=f"Testing"):
            img = data["img"].to(DEVICE).float()
            ori_label = data['ori_mask'].to(DEVICE)
            mask_name = data['mask_name'][0]

            out = net(img)
            out = sigmoid(out)
            out = torch.nn.functional.interpolate(
                out,
                [ori_label.shape[1], ori_label.shape[2]],
                mode='bilinear',
                align_corners=False
            )

            pred = (out.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(save_path / mask_name), pred)

    print("------- Testing Done -------")
    return save_path

if __name__ == "__main__":
    # This is an example of how to run the test script.
    # You would typically call this after training a model.
    # For example, to test the model from epoch 80:
    # test(CHECKPOINT_ROOT / f"{MODEL_NAME}-epoch-80.pth", epoch=80)

    # You can manually specify a model path to test here.
    # For example:
    # model_to_test = "/path/to/your/model.pth"
    # test(model_to_test, epoch=<your_epoch_number>)

    print("This script is intended to be called with a specific model path.")
    print("Please modify the __main__ block to test a trained model.")
