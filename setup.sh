#!/bin/bash
set -e

mkdir -p /kaggle/temp/datasets/DUTS
cd /kaggle/temp/datasets/DUTS
mkdir Image_train
mkdir Image_test
mkdir GT_train
mkdir GT_test

cp /kaggle/input/duts-saliency-detection-dataset/DUTS-TR/DUTS-TR-Image/* Image_train/
cp /kaggle/input/duts-saliency-detection-dataset/DUTS-TR/DUTS-TR-Mask/* GT_train/
cp /kaggle/input/duts-saliency-detection-dataset/DUTS-TE/DUTS-TE-Image/* Image_test/
cp /kaggle/input/duts-saliency-detection-dataset/DUTS-TE/DUTS-TE-Mask/* GT_test/

echo "DUTS dataset has been successfully copied to /kaggle/temp/datasets/DUTS"
