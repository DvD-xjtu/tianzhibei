# Final v3 数据下载与解压

Final v3 含 **21,500 张训练图、1,500 张合成测试图**及前景实例掩膜。冻结 crop 池分为训练侧 4,933 个、测试侧 5,156 个，保留透明前景和原始裁片。完整类别、来源、使用次数和隔离统计见 [Final v3 报告](docs/Final_v3_15类数据与crop剩余统计_2026-10-03.md)。

`data/` 有 7 个 Git LFS tar 分卷，总计约 **5.87 GiB**。训练图按文件名分成 5 卷；`dataset_other.tar` 存放测试图、训练/测试标签、逐图掩膜及数据集元信息；`crop_cases.tar` 存放全部冻结 crop 和对应 manifest。每卷均小于 2 GiB。`data/SHA256SUMS` 给出校验值。

下载与解压：

```bash
git clone https://github.com/DvD-xjtu/tianzhibei.git
cd tianzhibei
git lfs pull
sha256sum -c data/SHA256SUMS
mkdir -p final_v3_data
for archive in data/*.tar; do tar -xf "$archive" -C final_v3_data; done
```

解压后，`final_v3_data/dataset/` 中的 `dataset.yaml`、`train/`、`test/`、`instance_masks/` 可直接读取；`final_v3_data/crops/` 包含按训练/测试及类别划分的透明 crop、原始裁片和 `manifest_train.jsonl`、`manifest_test.jsonl`。所有合成图都是无可见框的图像；只有 `dataset/preview/` 中的展板绘有框。

分卷里的图像、标签和掩膜已经从服务器原有的符号链接实化为真实文件。逐图索引中这三类路径相对于 `dataset/`；原始背景与中间 manifest 的服务器路径仅用于来源追溯，不随包提供。使用 YOLO 训练器时，如其路径解析规则不同，请将 `dataset/dataset.yaml` 的 `path` 改为解压后 `dataset/` 的绝对路径。

代码见 `code/`。打包脚本为 `code/package_final_v3_for_github.py`。本包是合成数据；YF-12A 标注使用 SR-71A 外形代理，F-35 和 E-2 的独立来源稀少。公开使用或再分发图像前，请核对各上游数据来源的许可条件。
