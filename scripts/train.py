import os
import sys
import shutil
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset.sod_dataset import getDefectDataloader
from model import DefectSAM
from utils.loss import LossFunc
from config import (
    DEVICE,
    MODEL_NAME,
    EPOCHS,
    EPOCHS_WARM,
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    WEIGHT_DECAY,
    IMG_SIZE,
    SAM_PRETRAINED_PATH,
    LOG_DIR,
    CHECKPOINT_ROOT,
    SAVE_ROOT,
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
)

def main():
    """
    Main training loop for the DefectSAM model.
    """
    # ------- 1. Model setup -------
    print("--- Setting up model ---")
    net = DefectSAM(IMG_SIZE)

    # Note: Loading of SAM weights can be added here if needed.
    # For example:
    # load_sam(net, SAM_PRETRAINED_PATH, IMG_SIZE)

    # ------- 2. Optimizer setup -------
    print("--- Setting up optimizer ---")
    # Separate parameters for different learning rates
    param_last = []
    param_new = []
    for k, v in net.named_parameters():
        if 'image_encoder' in k:
            if 'adapter' in k:
                param_new.append(v)
            elif 'neck' in k:
                param_last.append(v)
            else:
                v.requires_grad = False
        else:
            if any(x in k for x in ['transformer', 'iou_token', 'output_upscaling']):
                param_last.append(v)
            else:
                param_new.append(v)

    optimizer = torch.optim.AdamW(
        [
            {"params": param_new, "lr": LR},
            {"params": param_last, "lr": LR * 0.1}
        ],
        weight_decay=WEIGHT_DECAY
    )

    # ------- 3. Dataloader setup -------
    print("--- Setting up dataloader ---")
    # The config for the dataloader needs to be an object with attributes.
    # We create a simple object for this.
    class DataloaderConfig:
        def __init__(self):
            self.mode = "train"
            self.img_dir = TRAIN_IMG_DIR
            self.mask_dir = TRAIN_MASK_DIR
            self.img_ext = ".bmp"
            self.mask_ext = ".png"
            self.batch_size = BATCH_SIZE
            self.num_workers = NUM_WORKERS

    dataloader_cfg = DataloaderConfig()
    dataloader_train = getDefectDataloader(dataloader_cfg, img_size=IMG_SIZE)

    # ------- 4. Loss function setup -------
    loss_func = LossFunc

    # ------- 5. Start training -------
    print("------- Start Training -------")
    net.to(DEVICE)
    writer = SummaryWriter(LOG_DIR)

    for epoch in range(1, EPOCHS + 1):
        print(f"--- Starting Epoch {epoch}/{EPOCHS} ---")

        # Learning rate scheduling
        if epoch <= EPOCHS_WARM:
            lr_epoch = LR * epoch / EPOCHS_WARM
        else:
            lr_epoch = LR * (0.98 ** (epoch - EPOCHS_WARM))

        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_epoch if i == 0 else lr_epoch * 0.1

        net.train()
        total_loss = 0
        for i, data in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch}")):
            img = data['img'].to(DEVICE, non_blocking=True).float()
            mask = data['mask'].to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            pred = net(img)
            loss = loss_func(pred, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"[Epoch: {epoch}/{EPOCHS}, Batch: {i+1}/{len(dataloader_train)}] Train Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader_train)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"--- Epoch {epoch} Summary ---")
        print(f"Average Training Loss: {avg_loss:.4f}")

        # ------- 6. Save checkpoint -------
        checkpoint_path = CHECKPOINT_ROOT / f"{MODEL_NAME}-epoch-{epoch}.pth"
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("------- Training Done -------")

if __name__ == "__main__":
    main()
