"""
Configuration file for the MDSAM project.
"""
from pathlib import Path

#
# <<< General settings >>>
#

# Device to use for training and inference.
DEVICE = "cuda"

# Name of the model to use.
MODEL_NAME = "DefectSAM"

#
# <<< Data settings >>>
#

# Root directory of the datasets.
DATASET_ROOT = Path('../datasets/')

# Name of the dataset to use for training.
TRAIN_DATASET = "NEU-RSDDS-AUG"

# Name of the dataset to use for testing.
TEST_DATASET = "NEU-RSDDS-AUG"

# Image and mask file extensions.
IMG_EXT = ".bmp"
MASK_EXT = ".png"

#
# <<< Training settings >>>
#

# Number of epochs to train for.
EPOCHS = 80

# Number of warm-up epochs.
EPOCHS_WARM = 5

# Batch size for training.
BATCH_SIZE = 4 # As per user request, set to a small value.

# Number of workers for the dataloader.
NUM_WORKERS = 0 # As per user request, set to a small value.

# Learning rate.
LR = 5e-4 * 0.5

# Weight decay for the optimizer.
WEIGHT_DECAY = 1e-4

# Image size for training.
IMG_SIZE = 384

#
# <<< Paths >>>
#

# Path to the pretrained SAM model.
SAM_PRETRAINED_PATH = "./pretrained/sam_vit_b_01ec64.pth"

# Directory to save TensorBoard logs.
LOG_DIR = Path('./logs')

# Directory to save model checkpoints.
CHECKPOINT_ROOT = Path('/hy-tmp/ckpt')

# Directory to save output results.
SAVE_ROOT = Path('/hy-tmp/output')

#
# <<< Derived paths (do not change) >>>
#

# Path to the training dataset.
TRAIN_DATASET_PATH = DATASET_ROOT / TRAIN_DATASET

# Path to the test dataset.
TEST_DATASET_PATH = DATASET_ROOT / TEST_DATASET

# Path to the training images.
TRAIN_IMG_DIR = TRAIN_DATASET_PATH / f'Image_train'
TRAIN_MASK_DIR = TRAIN_DATASET_PATH / f'GT_train'

# Path to the testing images.
TEST_IMG_DIR = TEST_DATASET_PATH / f'Image_test'
TEST_MASK_DIR = TEST_DATASET_PATH / f'GT_test'

# Create directories if they don't exist.
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
