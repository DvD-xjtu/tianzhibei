# Copy-Paste Mar-20 v1：LAE-1M 六子集小批量

日期：2026-09-27

## 本轮范围

- 前景：Mar-20 train 中经 SAM2.1 抠图、自动质量门通过的 401 个前景；没有人工或多模态视觉筛选。
- 前景增强：每个基础前景最多生成 4 个变体；旋转 0–360°、缩放 0.8–1.2、亮度 ±3、对比度 0.97–1.03、高斯噪声标准差 0–1.2。变换后的 alpha mask 再过连通域/面积门。本轮生成 1,599 个变体，5 个变体被 mask 门拒绝。
- 背景：仅从 LAE-1M 六个子集的 train 标注图中取已有飞机目标的场景；FAIR1M 仅取 `train_` 图，不读 valid/test。
- 每个子集生成 100 张候选，共 600 张。每个子集按 Mar-20 的 20 类均衡生成，每类 5 张。
- 自动过滤：沿用 `engineering-v7-1` 的背景泄漏、mask 碎片、放置区域纹理/植被/边缘/颜色，以及前景对比度和光度/饱和度差规则；没有人工筛图。

## 工程通过率

| LAE-1M 子集 | 候选 | 工程通过 | 拒绝 | 通过率 |
|---|---:|---:|---:|---:|
| DIOR | 100 | 22 | 78 | 22.0% |
| DOTAv2 | 100 | 19 | 81 | 19.0% |
| FAIR1M | 100 | 19 | 81 | 19.0% |
| NWPU VHR-10 | 100 | 15 | 85 | 15.0% |
| RSOD | 100 | 21 | 79 | 21.0% |
| xView | 100 | 9 | 91 | 9.0% |
| **合计** | **600** | **105** | **495** | **17.5%** |

该通过率是既定程序门的结果，不等于人工确认可用。xView 通过率最低（9%），主要失败原因为 paste lightness mismatch；本轮没有为凑数量放宽阈值。

## 标签与训练状态

输出为 Mar-20 20 类预览标签：只写新增飞机 HBB，原 LAE 标注作为逐图 sidecar 保留，没有合并进统一训练标签，因此 `train_ready=false`。这些数据用于看板审阅，不应直接作为最终训练集。

## 展板与产物

8999 看板的旧 MTARSI 前景版本已归入一级 **copy-paste-mtarsi**，v9 显示为 **v9-mtarsi**；新结果独立归入 **copy-paste-mar-20**，含六子集筛选入口，只展示自动门通过图。

- 增强前景：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/foreground_mar20_aug_v2/`
- 600 张候选：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_mar20_v1_lae_candidates/<子集>/`
- 自动通过集：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_mar20_v1_lae_engineering_pass/<子集>/`
- 逐例判定：各子集 `qa/decisions.jsonl`；指标摘要：各子集 `qa/summary.json`
- 代码：[`augment_mar20_foregrounds_v2.py`](../code/augment_mar20_foregrounds_v2.py)、[`synthesize_lae1m_mar20_v1.py`](../code/synthesize_lae1m_mar20_v1.py)、[`filter_lae1m_mar20_v1.py`](../code/filter_lae1m_mar20_v1.py)、[`register_mar20_lae_dashboard.py`](../code/register_mar20_lae_dashboard.py)
