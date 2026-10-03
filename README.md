# 遥感飞机 Copy-Paste 数据合成 Pipeline

本仓库保存飞机前景提取、质量筛选、遥感背景粘贴、数据集注册与审计的代码和工作文档。当前成果为 **Final v3（2026-10-03）**：21,500 张合成训练图、1,500 张合成测试图，目标类别为 15 类；完整统计见 [Final v3 报告](docs/Final_v3_15类数据与crop剩余统计_2026-10-03.md)。YF-12A 使用 SR-71A 外形代理，F-35 和 E-2 为稀少类。

可下载数据的分卷、校验和解压方法见 [数据包说明](DATASET_DOWNLOAD.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `code/` | 实际批量处理脚本：前景提取、分割与筛选、copy-paste、Final v3 生成、注册和审计 |
| `src/` | 早期 SAM2 基准测试与预览脚本 |
| `docs/` | 设计、阶段实验与 Final v3 统计报告；早期方案保留为历史记录 |

Final v3 的主要入口和依赖关系：

1. `code/extract_mtarsi_foregrounds.py`、`code/extract_mar20_foregrounds.py` 及各补充来源提取脚本制作原始前景池；`code/refine_extra_masks.py`、`code/filter_mar20_foregrounds.py` 等脚本筛选透明 crop。
2. `code/synthesize_copypaste_v6.py` 和 `code/produce_extra_engine.py` 实现尺度、空位、明暗、碰撞和质量门；`code/produce_final_v3_mixed.py` 生成基础训练批次。
3. `code/produce_f35_v3_supplement.py`、`code/produce_sr71_v3_supplement.py` 生成训练增量；`code/produce_final_v3_test.py` 用隔离的 crop 池生成合成测试批次。
4. `code/register_v3_test_combined.py` 注册统一目录；`code/audit_v3_split_isolation.py`、`code/report_final_v3_inventory.py` 检查隔离与统计。

脚本使用 Python 3，主要依赖 NumPy、Pillow、OpenCV、PyTorch、Requests 和 SAM2。SAM2 是外部项目，按其官方说明安装；本仓库不收录虚拟环境、模型权重或第三方源码。

## 数据与运行条件

本仓库的数据分卷包含最终合成图、标签、掩膜和冻结 crop 池；**不包含上游原始遥感数据或模型权重**。脚本是本项目实际生成批次时使用的版本，部分输入根目录写在 `code/produce_final_dataset.py`、`code/synthesize_lae1m_v9.py` 等文件的常量中；在其他机器重新生成时，需要安装依赖、准备相同的数据来源与 manifest，并按本机目录调整这些常量。各阶段所需文件和数据口径可从对应脚本及 `docs/` 的阶段报告查阅。

服务器上的成品数据目录为 `/data3/tianzhibei/derived/final_v3_20261003/combined_21500_train_1500_test`，其自带 README、`dataset.yaml`、类别表和逐图索引。该目录采用符号链接组织图像与掩膜，复制数据时必须连同链接目标一起打包；直接复制目录结构会产生失效链接。合成测试集用于检查相同 copy-paste 流程的泛化表现，不能替代独立真实场景测试集。

公开分发任何图像或 crop 前，应分别核对上游数据来源的再分发条款。代码仓库不授予对上游图像的再分发许可。
