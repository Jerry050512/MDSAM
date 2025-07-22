#!/bin/bash
set -e

python train.py

cd /hy-tmp

# 压缩包名称
file="train-result-$(date "+%Y%m%d-%H%M%S").zip"
# 把 result 目录做成 zip 压缩包
zip -q -r "${file}" output
# 通过 oss 上传到个人数据中的 results 文件夹中
oss cp "${file}" oss://results/
rm -f "${file}"

# 传输成功后关机
shutdown