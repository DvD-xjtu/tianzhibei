# Copy-Paste v9：LAE-1M 六子集实验

日期：2026-09-25

## 实验范围

本版将目标背景从 MAR20 更换为 LAE-1M 的六个 FOD 子集：DIOR、DOTAv2、FAIR1M、NWPU VHR-10、RSOD、xView。只从各子集的训练 split 取背景；FAIR1M 只使用 `train_` 图像，不使用 `valid_`。

前景来自 MTARSI-fixed 与 MTARSI-INNAR train 的已通过飞机抠图池，并使用现有几何增强池（旋转、0.8–1.2 缩放及轻微颜色/成像扰动）。当前与 MAR20 对齐的是 8 类：C-130、C-17、F-16、E-3、B-52、B-1B、F-15、F/A-18；不代表最初 15 类都已覆盖。

## 数量与工程过滤通过率

| LAE-1M 子集 | 候选合成图 | 工程通过 | 工程拒绝 | 通过率 |
|---|---:|---:|---:|---:|
| DIOR | 600 | 166 | 434 | 27.67% |
| DOTAv2 | 750 | 158 | 592 | 21.07% |
| FAIR1M | 700 | 170 | 530 | 24.29% |
| NWPU VHR-10 | 700 | 176 | 524 | 25.14% |
| RSOD | 600 | 176 | 424 | 29.33% |
| xView | 850 | 62 | 788 | 7.29% |
| **合计** | **4,200** | **908** | **3,292** | **21.62%** |

所有六个子集都生成了远多于 150 张的候选图。通过数不等于候选数；特别是 xView 仅有 62 张通过，未达到“每个子集至少 150 张工程通过图”的更严格目标。其主批通过率低，主要拒绝原因是粘贴区域亮度差不满足统一门槛；没有为凑数量放宽过滤条件。

工程通过不是人工审图结论。本轮未做人工或多模态视觉筛选；通过/拒绝完全由 `engineering-v7-1` 的颜色与亮度差、飞机前景对比度、mask 碎片/背景泄漏、放置区域纹理/边缘/植被等规则自动判定。通过只表示符合这些工程规则，仍需用户在看板上做最终视觉审核。

## 标签与训练状态

当前合成标签是用于预览的 MAR20 8 类 HBB 标签；LAE 原始标注按图另存，不与合成飞机类别强行合并。每条样本保留背景数据集、训练 split、原标注路径及哈希等溯源信息。因此本批数据标记为 `train_ready=false`：不能直接当作最终 LAE-1M 训练集。后续需先决定 LAE 原类标注与新增机型类别的映射/共存策略，再冻结正式训练标签。

## 可视化看板

8999 看板的“🔬 copy-paste”组已登记每个子集的候选对照图和工程通过对照图；候选视图包含被过滤的案例，工程通过视图仅含自动门槛通过案例。NWPU、DOTAv2、FAIR1M、xView 的补充批次也单独登记，方便核对。对照图左侧为原背景、右侧为合成结果；所有条目 `training=false`。

看板地址：<http://127.0.0.1:8999>

## 产物位置

- 候选图：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v9_lae1m_150pass_candidates/<子集>/`
- 补充候选：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v9_lae1m_topup_candidates/<子集>/`
- 工程通过集及逐图决定：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v9_lae1m_150pass_engineering_pass/<子集>/`
- 生成代码：[`synthesize_lae1m_v9.py`](../code/synthesize_lae1m_v9.py)、[`verify_lae1m_v9.py`](../code/verify_lae1m_v9.py)、[`filter_lae1m_v9.py`](../code/filter_lae1m_v9.py)

`qa/summary.json` 保存每个子集的通过率与类别通过数，`qa/decisions.jsonl` 保存逐图自动指标和拒绝原因。
